"""Generate a coarse trajectory from RGB-D files on CPU, without Isaac Sim or USD."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from vision.rgbd import RGBDCalibration, RGBDSettings, generate_rgbd_trajectory, save_rgbd_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True, help="Registered Z depth: .npy or single-channel PNG")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rgbd_offline"))
    parser.add_argument("--spacing", type=float, default=0.01)
    parser.add_argument("--clearance", type=float, default=0.01)
    parser.add_argument("--path-mode", choices=("auto", "closed", "open"), default="auto")
    args = parser.parse_args()
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    depth = np.load(args.depth, allow_pickle=False) if args.depth.suffix == ".npy" else np.asarray(Image.open(args.depth))
    result = generate_rgbd_trajectory(rgb, depth, RGBDCalibration.load(args.calibration),
                                      RGBDSettings(waypoint_spacing_m=args.spacing, tcp_offset_m=args.clearance,
                                                   path_mode=args.path_mode))
    save_rgbd_trajectory(result, rgb, args.output_dir)
    print(f"Generated {len(result.path.poses)} base-frame poses from RGB-D: {args.output_dir}")


if __name__ == "__main__":
    main()
