"""Choose real arm/base poses that frame a complete T-shaped CAD workpiece.

Only geometry, optics and static collision checks select views. Model predictions
and depth/position errors are never used in this selection.
"""

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from hj90.arm_kinematics import ArmKinematics


def world_matrix(cache, prim):
    return np.array(cache.GetLocalToWorldTransform(prim)).T


def transform_points(matrix, points):
    return (np.c_[points, np.ones(len(points))] @ matrix.T)[:, :3]


def look_at(eye, target):
    forward = np.asarray(target)-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    matrix[:3, 3] = eye
    return matrix


def t_geometry(stage, station):
    from pxr import UsdGeom
    fixture = stage.GetPrimAtPath('/World/Fixtures/'+station)
    if not fixture:
        raise ValueError('Missing station: '+station)
    cache = UsdGeom.XformCache()
    fixture_world = world_matrix(cache, fixture)
    bounds, vertices = {}, []
    for suffix in ('base', 'web'):
        prim = stage.GetPrimAtPath(str(fixture.GetPath())+'/'+station+'_'+suffix)
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError('Expected rectangular T parts: '+station)
        points = transform_points(np.linalg.inv(fixture_world) @ world_matrix(cache, prim),
                                  np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get()))
        low, high = points.min(0), points.max(0)
        if np.max(np.minimum(np.abs(points-low), np.abs(points-high))) > 1e-6:
            raise ValueError('T planner requires rectangular fixture-local parts')
        bounds[suffix] = np.stack((low, high))
        vertices.extend(points.tolist())
    base, web = bounds['base'], bounds['web']
    if abs(base[1, 2]-web[0, 2]) > 1e-6:
        raise ValueError('T web does not meet base top')
    # Visible-front surface samples, not weld annotations, for CAD ray tests.
    surfaces = [[x, y, base[1, 2]] for x in np.linspace(*base[:, 0], 9)
                for y in np.linspace(base[0, 1], web[0, 1]-.02, 4)]
    surfaces += [[x, web[0, 1], z] for x in np.linspace(*web[:, 0], 9)
                 for z in np.linspace(*web[:, 2], 5)]
    return fixture_world, bounds, np.array(vertices), np.array(surfaces)


def framing(pose, vertices, intrinsics, size, baseline=.09, margin=.04):
    scores, boxes = [], []
    for side in (-1, 1):
        camera = pose.copy()
        camera[:3, 3] += pose[:3, 0]*(side*baseline/2)
        points = transform_points(np.linalg.inv(camera), vertices)
        projected = points @ np.asarray(intrinsics).T
        xy = projected[:, :2]/projected[:, 2:]/np.asarray(size)
        scores.append(float(np.mean((points[:, 2] > 0) & (xy[:, 0] >= margin) &
                      (xy[:, 0] <= 1-margin) & (xy[:, 1] >= margin) & (xy[:, 1] <= 1-margin))))
        boxes.append([xy.min(0).tolist(), xy.max(0).tolist()])
    return min(scores), boxes


class OverviewPlanner:
    def __init__(self, stage, collision_file, calibration):
        import mujoco as mj
        from pxr import Usd, UsdGeom
        self.stage, self.calibration = stage, calibration
        self.arm = ArmKinematics(stage)
        self.source_base = world_matrix(UsdGeom.XformCache(), stage.GetPrimAtPath('/World/Robot/base_link'))
        self.inverse_base = np.linalg.inv(self.source_base)
        self.model = mj.MjModel.from_xml_path(str(Path(collision_file)))
        self.data = mj.MjData(self.model)
        self.arm_addresses = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot/Joints')):
            attr = prim.GetAttribute('state:angular:physics:position')
            index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, prim.GetName())
            if attr and index >= 0:
                self.data.qpos[self.model.jnt_qposadr[index]] = np.radians(attr.Get())
        for i in range(1, 7):
            index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, f'e05l_joint{i}')
            self.arm_addresses.append(self.model.jnt_qposadr[index])
        self.body = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'base_link')
        self.feet = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] != 0 and
                     self.model.geom_type[i] == mj.mjtGeom.mjGEOM_SPHERE and self.model.geom_size[i, 0] > .03]
        self.set_collision_pose(self.source_base, self.arm.initial)
        for path, expected in self.arm.forward(self.arm.initial).items():
            index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, path.split('/')[-1])
            if index >= 0:
                np.testing.assert_allclose(self.data.xpos[index], expected[:3, 3], atol=2e-5)
                np.testing.assert_allclose(self.data.xmat[index].reshape(3, 3), expected[:3, :3], atol=2e-5)

    def set_collision_pose(self, base, q):
        import mujoco as mj
        quat = Rotation.from_matrix(base[:3, :3]).as_quat()
        self.data.qpos[:3] = base[:3, 3]
        self.data.qpos[3:7] = np.r_[quat[3], quat[:3]]
        self.data.qpos[self.arm_addresses] = q
        mj.mj_forward(self.model, self.data)

    def solve(self, pose, yaw):
        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
        base[2, 3] = self.source_base[2, 3]
        target_rotation = base[:3, :3].T @ pose[:3, :3]
        target_z = pose[2, 3]-base[2, 3]

        def residual(q, regularize=False):
            camera = self.inverse_base @ self.arm.camera(q)
            error = np.r_[camera[2, 3]-target_z, .35*Rotation.from_matrix(
                target_rotation @ camera[:3, :3].T).as_rotvec()]
            return np.r_[error, .006*(q-self.arm.initial)] if regularize else error

        fit = least_squares(lambda q: residual(q, True), self.arm.initial,
                            bounds=self.arm.bounds.T, max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        exact = least_squares(residual, fit.x, bounds=self.arm.bounds.T, max_nfev=100,
                              ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if np.linalg.norm(residual(exact.x)) > 2e-5:
            return None
        camera = self.inverse_base @ self.arm.camera(exact.x)
        base[:2, 3] = pose[:2, 3]-(base[:3, :3] @ camera[:3, 3])[:2]
        return base, exact.x

    def choose(self, station):
        import mujoco as mj
        fixture, bounds, vertices, surface = t_geometry(self.stage, station)
        vertices, surface = transform_points(fixture, vertices), transform_points(fixture, surface)
        centre = np.array([0., 0., bounds['base'][1, 2]+.4*np.ptp(bounds['web'][:, 2])])
        records, best = [], None
        counts = dict(framed=0, ik=0, collision=0, floor=0, support=0)
        # Fixed geometric search; no model or truth-depth feedback.
        for azimuth in (0, -25, 25, -40, 40):
            for radius in (1.35, 1.55, 1.75):
                for elevation in (10, 15, 25, 35):
                    angle = np.radians(azimuth)
                    local_eye = centre+np.array([radius*np.sin(angle), -radius*np.cos(angle),
                                                radius*np.tan(np.radians(elevation))])
                    pose = look_at(transform_points(fixture, local_eye[None])[0],
                                   transform_points(fixture, centre[None])[0])
                    fraction, boxes = framing(pose, vertices, self.calibration['intrinsics'],
                                               self.calibration['image_size_wh'])
                    if fraction < 1:
                        continue
                    counts['framed'] += 1
                    yaw = np.arctan2(pose[1, 2], pose[0, 2])
                    solution = self.solve(pose, yaw)
                    if solution is None:
                        continue
                    counts['ik'] += 1
                    base, q = solution
                    self.set_collision_pose(base, q)
                    bad = [c for c in self.data.contact if c.dist < -.0005]
                    feet = self.data.geom_xpos[self.feet]
                    floor_error = np.max(np.abs(feet[:, 2]-self.model.geom_size[self.feet, 0]))
                    hull = ConvexHull(feet[:, :2])
                    com = self.data.subtree_com[self.body, :2]
                    support_margin = -float(np.max(hull.equations[:, :2] @ com+hull.equations[:, 2]))
                    counts['collision'] += bool(bad)
                    counts['floor'] += floor_error > .001
                    counts['support'] += support_margin < 0
                    if bad or floor_error > .001 or support_margin < 0:
                        continue
                    visibility = []
                    for sign in (-1, 1):
                        eye = pose[:3, 3]+sign*.045*pose[:3, 0]
                        count = 0
                        for point in surface:
                            ray = point-eye; distance = np.linalg.norm(ray); ray /= distance
                            geom = np.array([-1], dtype=np.int32)
                            hit = mj.mj_ray(self.model, self.data, eye, ray, None, 1, -1, geom)
                            count += hit >= 0 and abs(hit-distance) < .003
                        visibility.append(count/len(surface))
                    occupancy = min((box[1][0]-box[0][0])*(box[1][1]-box[0][1]) for box in boxes)
                    score = min(visibility)+.15*occupancy-.01*np.linalg.norm(q-self.arm.initial)
                    record = {'station': station, 'id': station+'_overview', 'azimuth_deg': azimuth,
                              'elevation_deg': elevation, 'horizontal_range_m': radius,
                              'arm_q_rad': q.tolist(), 'arm_q_deg': np.degrees(q).tolist(),
                              'base_world': base.tolist(), 'world_from_optical_midpoint': pose.tolist(),
                              'mesh_vertices_in_frame_both_eyes': fraction,
                              'mesh_normalized_bounds_both_eyes': boxes,
                              'front_surface_ray_visibility_both_eyes': visibility,
                              'foot_floor_max_error_m': float(floor_error), 'com_support_margin_m': support_margin,
                              'collision_penetration_limit_m': .0005, 'score': float(score)}
                    records.append(record)
                    if best is None or score > best['score']:
                        best = record
        if best is None:
            raise ValueError('No framed, supported, collision-free arm view for '+station+': '+str(counts))
        return best, records
