"""Persistent USD overlays, authored after vision capture to avoid contaminating it."""

import numpy as np
from pxr import UsdGeom


def draw_polyline(stage, name, points, color, width=0.002):
    points = np.asarray(points)
    if len(points) < 2:
        return
    curve = UsdGeom.BasisCurves.Define(stage, f"/World/Debug/{name}")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr(points.tolist())
    curve.CreateWidthsAttr([width])
    curve.SetWidthsInterpolation("constant")
    curve.CreateDisplayColorAttr([color])


def draw_plan(stage, path, world_from_path=None):
    transform = np.eye(4) if world_from_path is None else world_from_path
    surface_world = path.surface_points @ transform[:3, :3].T + transform[:3, 3]
    poses_world = transform[None] @ path.poses
    normal_world = -poses_world[0, :3, 2]
    draw_polyline(stage, "SurfaceContour", surface_world + 0.001 * normal_world, (0.0, 0.8, 0.1))
    draw_polyline(stage, "TCPReference", poses_world[:, :3, 3], (0.0, 0.4, 1.0))
    points = UsdGeom.Points.Define(stage, "/World/Debug/Waypoints")
    points.CreatePointsAttr(poses_world[:, :3, 3].tolist())
    points.CreateWidthsAttr([0.004] * len(path.poses))
    points.CreateDisplayColorAttr([(0., 0.4, 1.)])
    first = UsdGeom.Sphere.Define(stage, "/World/Debug/FirstPoint")
    first.CreateRadiusAttr(0.007)
    first.AddTranslateOp().Set(tuple(surface_world[0] + 0.01 * normal_world))
    first.CreateDisplayColorAttr([(1., 0.1, 0.)])
