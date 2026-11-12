"""Exact finite inner-face contacts of A04's three rectangular CAD parts."""

import numpy as np


def contact_segments(bounds):
    """Three welds specified by the user, in the fixture's local coordinates."""
    base, back, returned = [np.asarray(bounds[name], dtype=float) for name in ('base', 'back', 'return')]
    for box in (base, back, returned):
        if box.shape != (2, 3) or not np.isfinite(box).all() or not np.all(box[1] > box[0]):
            raise ValueError('Invalid rectangular CAD bounds')
    x, y, z = returned[1, 0], back[0, 1], base[1, 2]
    if not (back[0, 2]-.000001 <= z < back[1, 2] and returned[0, 2]-.000001 <= z < returned[1, 2]):
        raise ValueError('Upright plates do not meet base top')
    back_end = min(base[1, 0], back[1, 0])
    return_end = max(base[0, 1], returned[0, 1])
    top = min(back[1, 2], returned[1, 2])
    if not (base[0, 0] <= x < back_end and return_end < y <= base[1, 1] and
            back[0, 0] <= x <= back[1, 0] and abs(returned[1, 1]-y) < 1e-6):
        raise ValueError('A04 inner contact topology is inconsistent')
    junction = np.array([x, y, z])
    return {'A04_vertical': np.stack((junction, [x, y, top])),
            'A04_back_base': np.stack((junction, [back_end, y, z])),
            'A04_return_base': np.stack((junction, [x, return_end, z]))}


def reference_from_stage(stage):
    from pxr import Usd, UsdGeom
    fixture = stage.GetPrimAtPath('/World/Fixtures/A04')
    cache = UsdGeom.XformCache()
    transform = np.array(cache.GetLocalToWorldTransform(fixture)).T
    inverse = np.linalg.inv(transform)
    bounds = {}
    for prim in Usd.PrimRange(fixture):
        if prim.GetName() not in ('A04_base', 'A04_back', 'A04_return') or not prim.IsA(UsdGeom.Mesh):
            continue
        vertices = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        local = inverse @ np.array(cache.GetLocalToWorldTransform(prim)).T
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ local.T)[:, :3]
        low, high = vertices.min(0), vertices.max(0)
        if np.max(np.minimum(np.abs(vertices-low), np.abs(vertices-high))) > 1e-6:
            raise ValueError('Reference requires axis-aligned rectangular parts in fixture frame')
        bounds[prim.GetName()[4:]] = np.stack((low, high))
    if set(bounds) != {'base', 'back', 'return'}:
        raise ValueError('Missing A04 CAD parts')
    records = []
    for name, local in contact_segments(bounds).items():
        world = (np.c_[local, np.ones(2)] @ transform.T)[:, :3]
        records.append({'name': name, 'control_world_m': world, 'control_fixture_m': local,
                        'definition': 'Finite inner-face CAD intersection; all three are welds by user instruction'})
    return records
