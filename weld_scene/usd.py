"""Attach two independent cameras and a shadow-casting RTX laser projector."""

from pathlib import Path

import numpy as np

SOURCE_CAMERA = '/World/Robot/e05l_welding_torch/hj90'
RIG_PATH = '/World/Robot/e05l_welding_torch/hj90_active'
RGB_CAMERA = RIG_PATH + '/rgb'


def set_intrinsics(camera, intrinsics, size):
    """Author a USD frustum from edge-origin K, including a noncentral principal point."""
    from pxr import Gf
    k = np.asarray(intrinsics)
    w, h = size
    vertical = 24.
    horizontal = vertical*w*k[1, 1]/(h*k[0, 0])
    camera.CreateVerticalApertureAttr(vertical)
    camera.CreateHorizontalApertureAttr(horizontal)
    camera.CreateFocalLengthAttr(vertical*k[1, 1]/h)
    camera.CreateHorizontalApertureOffsetAttr(horizontal*(.5-k[0, 2]/w))
    camera.CreateVerticalApertureOffsetAttr(vertical*(k[1, 2]/h-.5))
    camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 100.))


def camera_intrinsics(camera, size):
    """Read edge-origin K from USD; offsets have opposite signs in image X and Y."""
    w, h = size
    focal = float(camera.GetFocalLengthAttr().Get())
    horizontal = float(camera.GetHorizontalApertureAttr().Get())
    vertical = float(camera.GetVerticalApertureAttr().Get())
    return np.array([[w*focal/horizontal, 0, w*(.5-camera.GetHorizontalApertureOffsetAttr().Get()/horizontal)],
                     [0, h*focal/vertical, h*(.5+camera.GetVerticalApertureOffsetAttr().Get()/vertical)],
                     [0, 0, 1.]])


def laser_texture(path, calibration):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A numerical angular emission profile, not a painted stripe on the target.
    width = 2048
    x = (np.arange(width)+.5-width/2) / (width/2)
    profile = np.exp(-.5 * (x/calibration['laser_sigma_rad'])**2)
    values = np.broadcast_to((profile*255).astype(np.uint8)[None, :, None], (16, width, 3)).copy()
    Image.fromarray(values).save(path)
    return path.resolve()


def attach(stage, calibration, texture, source_camera=SOURCE_CAMERA, rig_path=RIG_PATH, rgb_calibration=None):
    from pxr import Gf, Sdf, UsdGeom, UsdLux
    from .vendor_calibration import load_rgb_calibration, rgb_pose_in_midpoint
    source = UsdGeom.Camera(stage.GetPrimAtPath(source_camera))
    if not source:
        raise ValueError('HJ-90 reference camera is missing')
    rig = UsdGeom.Xform.Define(stage, rig_path)
    # Same parent torch: camera-to-tool relation persists as the robot moves.
    if str(source.GetPath().GetParentPath()) != str(rig.GetPath().GetParentPath()):
        raise ValueError('Rig and reference camera must share the torch parent')
    local = UsdGeom.Xformable(source).GetLocalTransformation()
    rig.MakeMatrixXform().Set(local)
    k = np.asarray(calibration['intrinsics'])
    w, h = calibration['image_size_wh']
    cameras = {}
    for side, x in (('left', -calibration['baseline_m']/2), ('right', calibration['baseline_m']/2)):
        camera = UsdGeom.Camera.Define(stage, rig_path+'/'+side)
        set_intrinsics(camera, k, (w, h))
        transform = Gf.Matrix4d(1.)
        transform.SetTranslate(Gf.Vec3d(x, 0, 0))
        camera.MakeMatrixXform().Set(transform)
        cameras[side] = str(camera.GetPath())
    # The SDK identifies a separate RGB sensor. Its measured K/T must never
    # overwrite the two unmeasured stereo eyes. Keep the stereo return map binary.
    rgb_calibration = rgb_calibration if rgb_calibration is not None else load_rgb_calibration()
    rgb = UsdGeom.Camera.Define(stage, rig_path+'/rgb')
    set_intrinsics(rgb, rgb_calibration['intrinsics'], rgb_calibration['image_size_wh'])
    optical_to_usd = np.diag([1., -1., -1., 1.])
    rgb_local = optical_to_usd @ rgb_pose_in_midpoint(rgb_calibration, calibration['baseline_m']) @ optical_to_usd
    rgb.MakeMatrixXform().Set(Gf.Matrix4d(rgb_local.T.tolist()))
    rgb.GetPrim().CreateAttribute('hj90:imageSizeWH', Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(*rgb_calibration['image_size_wh']))
    rgb.GetPrim().CreateAttribute('hj90:calibration', Sdf.ValueTypeNames.String).Set(__import__('json').dumps(rgb_calibration))
    laser = UsdLux.RectLight.Define(stage, rig_path+'/laser')
    prim = laser.GetPrim()
    prim.CreateAttribute('isProjector', Sdf.ValueTypeNames.Bool).Set(True)
    laser.CreateWidthAttr(2.)
    laser.CreateHeightAttr(2*np.tan(np.radians(calibration['laser_fan_half_angle_deg'])))
    laser.CreateTextureFileAttr(Sdf.AssetPath(str(Path(texture).resolve())))
    laser.CreateColorAttr(Gf.Vec3f(*calibration['laser_rgb']))
    laser.CreateIntensityAttr(calibration['laser_intensity'])
    laser.CreateNormalizeAttr(False)
    origin = calibration['laser_origin_optical_m']
    laser.AddTranslateOp().Set(Gf.Vec3d(origin[0], -origin[1], -origin[2]))
    laser.AddRotateYOp().Set(0.)
    prim.CreateAttribute('hj90:wavelengthNm', Sdf.ValueTypeNames.Float).Set(calibration['laser_wavelength_nm'])
    prim.CreateAttribute('hj90:status', Sdf.ValueTypeNames.String).Set(
        'RGB radiometric approximation of 450 nm; unknown power/divergence; not coherent wave optics')
    rig.GetPrim().CreateAttribute('hj90:calibration', Sdf.ValueTypeNames.String).Set(__import__('json').dumps(calibration))
    return cameras, laser, rig


def set_laser(laser, angle_rad, enabled, calibration):
    laser.GetPrim().GetAttribute('xformOp:rotateY').Set(-float(np.degrees(angle_rad)))
    laser.GetIntensityAttr().Set(calibration['laser_intensity'] if enabled else 0.)


def export_layer(bundle, calibration, base_layer=None):
    """Write a portable stronger USD layer; keep the original USDC unchanged."""
    import json
    import shutil
    from pxr import Sdf, Usd, UsdGeom
    from .vendor_calibration import DEFAULT_RGB_CALIBRATION, load_rgb_calibration
    bundle = Path(bundle)
    if base_layer is None:
        manifest_path = bundle/'manifest.json'
        if manifest_path.is_file():
            scene = json.loads(manifest_path.read_text())['full_scene']
            base_layer = scene['poses'][scene['default_pose']]['asset']
        else:
            base_layer = 'industrial_welding.usdc'
    base = bundle/base_layer
    if not base.resolve().is_relative_to(bundle.resolve()) or not base.is_file():
        raise ValueError('Missing or external sensor base layer: '+str(base_layer))
    name = 'industrial_welding_hj90.usda'
    output = bundle/name
    layer = Sdf.Layer.CreateNew(str(output)) if not output.exists() else Sdf.Layer.FindOrOpen(str(output))
    layer.Clear()
    layer.subLayerPaths = [base_layer]
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    texture = laser_texture(bundle/'textures/hj90_laser_profile.png', calibration)
    rgb_calibration = load_rgb_calibration()
    cameras, laser, _ = attach(stage, calibration, texture, rgb_calibration=rgb_calibration)
    laser.GetTextureFileAttr().Set(Sdf.AssetPath('textures/hj90_laser_profile.png'))
    layer.Save()
    output.write_text(output.read_text().rstrip()+'\n')
    (bundle/'source/hj90_sim_calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
    shutil.copyfile(DEFAULT_RGB_CALIBRATION, bundle/'source/hj90_rgb_vendor.json')
    full = bundle/'industrial_welding_full.usda'
    full.write_text('''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [@industrial_welding_hj90.usda@]
)
''')
    return {'layer': name, 'cameras': cameras, 'laser': str(laser.GetPath()),
            'calibration': 'source/hj90_sim_calibration.json',
            'rgb_camera': RGB_CAMERA,
            'rgb_calibration': {'file': 'source/hj90_rgb_vendor.json', **rgb_calibration},
            'extra_files': [name, 'textures/hj90_laser_profile.png', 'source/hj90_sim_calibration.json',
                            'source/hj90_rgb_vendor.json']}
