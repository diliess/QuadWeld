"""Score an auto-weld grid run against the environment manifest; evaluation only, never used to plan.

    python scripts/evaluate_grid_weld.py outputs/runs/b2_e10l_grid
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def true_corner(parts, seam_world):
    """The web-face / flange-top corner line nearest the observed seam, for axis-aligned parts."""
    centre = seam_world.mean(0)
    best = None
    for web in (p for p in parts if p["kind"] == "web"):
        c, s = np.array(web["center_m"]), np.array(web["size_m"])
        axis = int(np.argmin(s[:2]))  # Horizontal axis through the plate thickness.
        along = 1 - axis
        if abs(centre[along] - c[along]) > s[along] / 2:
            continue
        side = np.sign(centre[axis] - c[axis])
        face = c[axis] + side * s[axis] / 2
        if best is None or abs(centre[axis] - face) < best[0]:
            best = (abs(centre[axis] - face), web, axis, face, side)
    if best is None:
        raise ValueError("No web spans the observed seam")
    _, web, axis, face, side = best
    flanges = [p for p in parts if p["kind"] == "flange"
               and abs(np.array(p["center_m"])[axis] - np.array(web["center_m"])[axis]) < 1e-3]
    top = max(np.array(p["center_m"])[2] + np.array(p["size_m"])[2] / 2 for p in flanges)
    normals = np.zeros((2, 3))
    normals[0, axis] = side  # Web face normal, into the cell.
    normals[1, 2] = 1.       # Flange top.
    point = np.zeros(3)
    point[axis], point[2] = face, top
    return {"web": web["name"], "axis": "xy"[axis], "point": point, "normals": normals,
            "direction": np.eye(3)[1 - axis]}


def to_world(world_from_base, points):
    return np.asarray(points) @ world_from_base[:3, :3].T + world_from_base[:3, 3]


def offsets(points, corner):
    """Signed distances of points from the web face and from the flange top (positive: free space)."""
    return (points - corner["point"]) @ corner["normals"].T


def summary(values):
    values = np.asarray(values)
    return {"mean_m": float(values.mean()), "min_m": float(values.min()), "max_m": float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--environment-manifest", type=Path, default=ROOT / "assets/scenes/ship-grid/manifest.json")
    args = parser.parse_args()
    plan = np.load(args.run_dir / "trajectory.npz")
    world_from_base = plan["world_from_base"]
    parts = json.loads(args.environment_manifest.read_text())["parts"]
    seam = to_world(world_from_base, plan["seam_base"])
    corner = true_corner(parts, seam)
    seam_offsets = offsets(seam, corner)
    weld = to_world(world_from_base, plan["surface_points"])
    seam_report = json.loads((args.run_dir / "seam_report.json").read_text())
    result = {
        "true_corner": {"web": corner["web"], "seam_axis": "y" if corner["axis"] == "x" else "x",
                        "point_world_m": corner["point"].tolist()},
        "perceived_seam": {"samples": len(seam), "length_m": float(np.linalg.norm(np.diff(seam, axis=0), axis=1).sum()),
                           "offset_from_web_face": summary(seam_offsets[:, 0]),
                           "offset_from_flange_top": summary(seam_offsets[:, 1]),
                           "distance_to_corner_line": summary(np.linalg.norm(seam_offsets, axis=1))},
        "welded_segment": {"length_m": float(np.linalg.norm(weld[-1] - weld[0])),
                           "distance_to_corner_line": summary(np.linalg.norm(offsets(weld, corner), axis=1))},
        "depth_scale": seam_report["depth_scale"],
        "support_plane_angle_error_deg": seam_report["support_plane_angle_error_deg"],
    }
    execution = args.run_dir / "tcp_execution.csv"
    if execution.is_file():
        with execution.open() as stream:
            rows = [row for row in csv.DictReader(stream) if row["phase"] == "trace"]
        if rows:
            actual = to_world(world_from_base, [[float(r[k]) for k in ("actual_x", "actual_y", "actual_z")] for r in rows])
            gaps = offsets(actual, corner)
            nominal = json.loads((args.run_dir / "robot_config.json").read_text())["tcp_offset_m"]
            result["executed_trace"] = {
                "samples": len(actual),
                "nominal_tcp_distance_to_corner_m": nominal,
                "tcp_clearance_to_web_face": summary(gaps[:, 0]),
                "tcp_clearance_to_flange_top": summary(gaps[:, 1]),
                "tcp_distance_to_corner_line": summary(np.linalg.norm(gaps, axis=1)),
                "touched_a_face": bool(gaps.min() <= 0.),
            }
    (args.run_dir / "grid_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
