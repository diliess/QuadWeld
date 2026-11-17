"""Plan and validate the camera waypoints for a two-distance weld scan."""

import hashlib

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .mount_offset import apply_collision_offset, apply_usd_offset, torch_frustum_audit
from .optics import load_calibration
from .overview_planning import OverviewPlanner, look_at, static_check
from .scene import ROOT, STEREO_SCENE
from .stereo_capture import joint_geometry, rig_pose

COLLISION_MODEL = (ROOT.parent / 'robosimstudio/examples/live/twins/industrial_welding/assets/assembled'
                   / 'industrial_b2_e05l_pro.xml')
STATIONS = ('A25', 'A26')
MOUNT_OFFSET_M = .05


def calibration_and_planner(collision_file, stations):
    import mujoco as mj
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(STEREO_SCENE))
    stage.SetEditTarget(stage.GetSessionLayer())
    calibration = load_calibration()
    mount = apply_usd_offset(stage, MOUNT_OFFSET_M)
    if any(row['triangles_potentially_intersecting_frustum'] for row in torch_frustum_audit(stage, calibration)):
        raise ValueError('The torch intersects a stereo frustum')

    # Older MuJoCo scenes omit these two stations. Add bounds for the USD
    # workpiece, stand and clamps where the current collision scene lacks them.
    spec = mj.MjSpec.from_file(str(collision_file))
    existing = {geom.name for geom in spec.geoms}
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'proxy', 'render'], False, True)
    for station in stations:
        for prim in stage.GetPrimAtPath('/World/Fixtures/' + station).GetChildren():
            name = prim.GetName()
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or name in existing or '_bead_contact_' in name:
                continue
            box = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
            low, high = np.array(box.GetMin()), np.array(box.GetMax())
            spec.worldbody.add_geom(name=name, type=mj.mjtGeom.mjGEOM_BOX,
                                    pos=(low + high) / 2, size=(high - low) / 2)
    planner = OverviewPlanner(stage, spec.compile(), calibration)
    apply_collision_offset(planner.model, mount)
    return stage, planner


def arm_waypoint(planner, name, station, kind, pose, target, distance, **metadata):
    forward = pose[:3, 2]
    yaw = np.arctan2(forward[1], forward[0])
    for yaw_offset in (0, -20, 20, -45, 45):
        for reach in (.55, .65, .45):
            solution = planner.solve(pose, yaw + np.radians(yaw_offset), reach=reach, preserve_base_tilt=True)
            if solution is None:
                continue
            base, joints = solution
            valid, checks = static_check(planner, base, joints)
            if valid:
                return {'id': name, 'station': station, 'view': kind, 'joint_side': 'front',
                        'distance_m': distance, 'target_world_m': np.asarray(target).tolist(),
                        'world_from_optical_midpoint': pose.tolist(), 'base_world': base.tolist(),
                        'arm_q_rad': joints.tolist(), **metadata, **checks}
    raise ValueError(f'No collision-free, supported arm pose for {name}')


def transitions(planner, waypoints):
    """Audit the straight interpolation between independent capture poses."""
    records = []
    for first, second in zip(waypoints, waypoints[1:]):
        start, stop = (np.asarray(view['base_world']) for view in (first, second))
        qa, qb = (np.asarray(view['arm_q_rad']) for view in (first, second))
        length = np.linalg.norm(stop[:3, 3] - start[:3, 3])
        rotation = Rotation.from_matrix(np.stack((start[:3, :3], stop[:3, :3])))
        angle = (rotation[0].inv() * rotation[1]).magnitude()
        count = 1 + int(np.ceil(max(length / .03, angle / .04, np.max(np.abs(qb - qa)) / .04)))
        samples = np.linspace(0, 1, count)
        failed = 0
        for fraction, orient in zip(samples, Slerp([0, 1], rotation)(samples).as_matrix()):
            base = np.eye(4)
            base[:3, :3] = orient
            base[:3, 3] = (1 - fraction) * start[:3, 3] + fraction * stop[:3, 3]
            valid, _ = static_check(planner, base, (1 - fraction) * qa + fraction * qb)
            failed += not valid
        records.append({'from': first['id'], 'to': second['id'], 'static_samples': count,
                        'failed_samples': failed, 'base_distance_m': float(length)})
    return records


def plan_overview(stations, collision_model=COLLISION_MODEL):
    stage, planner = calibration_and_planner(collision_model, stations)
    views = []
    for station in stations:
        fixture, joint = joint_geometry(stage, station)
        pose = rig_pose(fixture, joint, 0., 2., 22., -15., -35., 'front')
        target = pose[:3, 3] + 2. * pose[:3, 2]
        views.append(arm_waypoint(planner, station + '_overview', station, 'overview', pose, target, 2.,
                                  elevation_deg=22., azimuth_deg=-15., roll_deg=-35.))
    return plan_record(collision_model, views)


def sweep_targets(ends, max_step=.15):
    """Aim left-to-right, with overlapping observations at both ends."""
    start, stop = np.asarray(ends, float)
    length = np.linalg.norm(stop - start)
    inset = min(.04 / length, .15)
    count = max(3, int(np.ceil(length * (1 - 2 * inset) / max_step)) + 1)
    if count % 2 == 0:
        count += 1
    return start + np.linspace(inset, 1 - inset, count)[:, None] * (stop - start)


def inspection_pose(target, normals, distance, elevation=35., roll=-35.):
    normals = np.asarray(normals)
    base = int(np.argmax(np.abs(normals[:, 2])))
    sine, cosine = np.sin(np.radians(elevation)), np.cos(np.radians(elevation))
    outward = sine * normals[base] + cosine * normals[1 - base]
    eye = target + distance * outward / np.linalg.norm(outward)
    pose = look_at(eye, target)
    c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    pose[:3, :3] = pose[:3, :3] @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return pose


def plan_close(overview, observations, collision_model=COLLISION_MODEL, *, range_checks=False):
    """Use measured world-frame roots and normals to place the near cameras."""
    _, planner = calibration_and_planner(collision_model, [v['station'] for v in overview])
    views, path_checks = [], []
    for view in overview:
        station = view['station']
        observation = observations[station]
        ends = np.asarray(observation['root_world_m'])
        normals = observation['face_normals_world']
        targets = sweep_targets(ends)
        sweep = []
        for i, target in enumerate(targets):
            pose = inspection_pose(target, normals, .6)
            sweep.append(arm_waypoint(planner, f'{station}_sweep_{i:02d}', station, 'sweep', pose, target, .6,
                                      scan_index=i, elevation_deg=35., roll_deg=-35.))
        path_checks.extend(transitions(planner, [view, *sweep]))
        views.extend(sweep)
        if not range_checks:
            continue
        target = targets[len(targets) // 2]
        for distance in (.45, .75):
            pose = inspection_pose(target, normals, distance)
            ident = f'{station}_range_{round(100 * distance):02d}'
            views.append(arm_waypoint(planner, ident, station, 'range_check', pose, target, distance,
                                      elevation_deg=35., roll_deg=-35.))
    return plan_record(collision_model, views, overview=observations, transitions=path_checks)


def plan_record(collision_model, views, **metadata):
    source = STEREO_SCENE.with_name('industrial_welding.usdc')
    return {'views': views, 'mount_offset_m': MOUNT_OFFSET_M, 'collision_model': str(collision_model),
            'source_usdc_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'physics_executed': False, **metadata}
