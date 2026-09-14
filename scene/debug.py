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


def draw_plan(stage, path):
    draw_polyline(stage, "SurfaceContour", path.surface_points + [0, 0, 0.001], (0.0, 0.8, 0.1))
    draw_polyline(stage, "TCPReference", path.poses[:, :3, 3], (0.0, 0.4, 1.0))
    points = UsdGeom.Points.Define(stage, "/World/Debug/Waypoints")
    points.CreatePointsAttr(path.poses[:, :3, 3].tolist())
    points.CreateWidthsAttr([0.004] * len(path.poses))
    points.CreateDisplayColorAttr([(0., 0.4, 1.)])
    first = UsdGeom.Sphere.Define(stage, "/World/Debug/FirstPoint")
    first.CreateRadiusAttr(0.007)
    first.AddTranslateOp().Set(tuple(path.surface_points[0] + [0, 0, 0.01]))
    first.CreateDisplayColorAttr([(1., 0.1, 0.)])
