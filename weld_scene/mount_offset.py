"""A reversible lateral bracket extension; no hidden geometry or RGB cropping."""

import numpy as np
from scipy.spatial.transform import Rotation

TOOL = '/World/Robot/e05l_welding_torch'


def movable(name):
    return name.startswith('hj90') and not name.startswith('hj90_collar') and name != 'hj90_bridge_plate'


def parallel_axis(position):
    return np.eye(3)*np.dot(position, position)-np.outer(position, position)


def shifted_inertia(mass, centre, tensor, payload_centre, delta, payload_mass=1.056):
    new_centre = centre+payload_mass/mass*delta
    new_tensor = (tensor+mass*parallel_axis(centre)+payload_mass*(
        parallel_axis(payload_centre+delta)-parallel_axis(payload_centre))-mass*parallel_axis(new_centre))
    values, axes = np.linalg.eigh(new_tensor)
    if values.min() <= 0:
        raise ValueError('Nonphysical payload inertia')
    if np.linalg.det(axes) < 0: axes[:, 0] *= -1
    return new_centre, values, axes


def apply_usd_offset(stage, offset_m):
    from pxr import Gf, UsdGeom, UsdPhysics
    if not .001 <= offset_m <= .15:
        raise ValueError('Bracket extension must be 1–150 mm')
    tool = stage.GetPrimAtPath(TOOL)
    camera = UsdGeom.Xformable(stage.GetPrimAtPath(TOOL+'/hj90'))
    old_optical = np.array(camera.GetLocalTransformation()).T @ np.diag([1., -1., -1., 1.])
    delta = old_optical[:3, 0]*offset_m
    np.testing.assert_allclose(delta, [0., -offset_m, 0.], atol=1e-10)
    payload = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(TOOL+'/hj90_envelope_collision')).GetLocalTransformation()).T[:3, 3]
    moved = []
    for prim in tool.GetChildren():
        if movable(prim.GetName()):
            transform = np.array(UsdGeom.Xformable(prim).GetLocalTransformation()).T
            transform[:3, 3] += delta
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(transform.T.tolist()))
            moved.append(prim.GetName())
    bridge = UsdGeom.Xformable(stage.GetPrimAtPath(TOOL+'/hj90_bridge_plate'))
    transform = np.array(bridge.GetLocalTransformation()).T
    transform[1, 1] += offset_m
    transform[:3, 3] += delta/2
    bridge.MakeMatrixXform().Set(Gf.Matrix4d(transform.T.tolist()))
    api = UsdPhysics.MassAPI(tool)
    mass = api.GetMassAttr().Get(); centre = np.array(api.GetCenterOfMassAttr().Get())
    quat = api.GetPrincipalAxesAttr().Get()
    rotation = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
    tensor = rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ rotation.T
    centre, values, axes = shifted_inertia(mass, centre, tensor, payload, delta)
    q = Rotation.from_matrix(axes).as_quat()
    api.GetCenterOfMassAttr().Set(Gf.Vec3f(*centre))
    api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*values))
    api.GetPrincipalAxesAttr().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
    return {'lateral_offset_m': offset_m, 'translation_in_torch_m': delta.tolist(),
            'old_torch_from_optical': old_optical.tolist(),
            'new_torch_from_optical': (old_optical+np.column_stack((np.zeros((4,3)),np.r_[delta,0.]))).tolist(),
            'moved_prims': moved, 'payload_original_centre_m': payload.tolist(),
            'bracket_mass_kg': None, 'status': 'Simulation bracket extension; not a manufactured/calibrated mount'}


def apply_collision_offset(model, record):
    import mujoco as mj
    delta = np.asarray(record['translation_in_torch_m'])
    for geom in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom) or ''
        if movable(name): model.geom_pos[geom] += delta
        if name == 'hj90_bridge_plate':
            model.geom_pos[geom] += delta/2
            model.geom_size[geom, 1] += record['lateral_offset_m']/2
            model.geom_aabb[geom, 3:] = model.geom_size[geom]
            model.geom_rbound[geom] = np.linalg.norm(model.geom_size[geom])
    body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'e05l_welding_torch')
    q = model.body_iquat[body]
    axes = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
    tensor = axes @ np.diag(model.body_inertia[body]) @ axes.T
    centre, values, axes = shifted_inertia(model.body_mass[body], model.body_ipos[body], tensor,
        np.asarray(record['payload_original_centre_m']), delta)
    q = Rotation.from_matrix(axes).as_quat()
    model.body_ipos[body], model.body_inertia[body], model.body_iquat[body] = centre, values, np.r_[q[3],q[:3]]


def torch_frustum_audit(stage, calibration):
    """Conservative triangle/frustum separating-plane test, including near plane.

    Zero potentially intersecting triangles certifies no torch geometry in either
    frustum. This is independent of robot joint or base pose.
    """
    from pxr import UsdGeom
    cache = UsdGeom.XformCache()
    world = lambda p: np.array(cache.GetLocalToWorldTransform(p)).T
    camera = world(stage.GetPrimAtPath(TOOL+'/hj90')) @ np.diag([1.,-1.,-1.,1.])
    fx, fy = calibration['intrinsics'][0][0], calibration['intrinsics'][1][1]
    cx, cy = calibration['intrinsics'][0][2], calibration['intrinsics'][1][2]
    width, height = calibration['image_size_wh']
    records = []
    for sign, side in ((-1,'left'), (1,'right')):
        eye = camera.copy(); eye[:3, 3] += sign*calibration['baseline_m']/2*camera[:3, 0]
        possible = triangles = 0
        for prim in stage.GetPrimAtPath(TOOL).GetChildren():
            if not prim.GetName().startswith('e05l_') or not prim.IsA(UsdGeom.Mesh): continue
            if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible': continue
            mesh = UsdGeom.Mesh(prim)
            points = np.array(mesh.GetPointsAttr().Get())
            xyz = (np.c_[points,np.ones(len(points))] @ (np.linalg.inv(eye) @ world(prim)).T)[:,:3]
            faces, start = [], 0
            ids = np.array(mesh.GetFaceVertexIndicesAttr().Get())
            for count in mesh.GetFaceVertexCountsAttr().Get():
                faces.extend((ids[start],ids[start+i],ids[start+i+1]) for i in range(1,count-1));start += count
            faces = np.array(faces); triangles += len(faces)
            x,y,z = xyz.T
            planes = np.stack((z-.005,100-z,fx*x+cx*z,-fx*x+(width-cx)*z,
                               fy*y+cy*z,-fy*y+(height-cy)*z),axis=1)
            possible += int((~(planes[faces].max(axis=1)<0).any(axis=1)).sum())
        records.append({'eye':side,'visible_torch_triangles_checked':triangles,
                        'triangles_potentially_intersecting_frustum':possible})
    return records
