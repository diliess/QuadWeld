"""Attach two independent cameras and a shadow-casting RTX laser projector."""

from pathlib import Path

import numpy as np

SOURCE_CAMERA = '/World/Robot/e05l_welding_torch/hj90'
RIG_PATH = '/World/Robot/e05l_welding_torch/hj90_active'


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


def attach(stage, calibration, texture, source_camera=SOURCE_CAMERA, rig_path=RIG_PATH):
    from pxr import Gf, Sdf, UsdGeom, UsdLux
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
        camera.CreateVerticalApertureAttr(24.)
        camera.CreateHorizontalApertureAttr(24*w*k[1, 1]/(h*k[0, 0]))
        camera.CreateFocalLengthAttr(24*k[1, 1]/h)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 100.))
        camera.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
        cameras[side] = str(camera.GetPath())
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


def export_layer(bundle, calibration):
    """Write a portable stronger USD layer; keep the original USDC unchanged."""
    import json
    from pxr import Sdf, Usd, UsdGeom
    bundle = Path(bundle)
    name = 'industrial_welding_hj90.usda'
    output = bundle/name
    layer = Sdf.Layer.CreateNew(str(output)) if not output.exists() else Sdf.Layer.FindOrOpen(str(output))
    layer.Clear()
    layer.subLayerPaths = ['industrial_welding_A04_initial_arm_photo.usda']
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    texture = laser_texture(bundle/'textures/hj90_laser_profile.png', calibration)
    cameras, laser, _ = attach(stage, calibration, texture)
    laser.GetTextureFileAttr().Set(Sdf.AssetPath('textures/hj90_laser_profile.png'))
    layer.Save()
    (bundle/'source/hj90_sim_calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
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
            'extra_files': [name, 'textures/hj90_laser_profile.png', 'source/hj90_sim_calibration.json']}
