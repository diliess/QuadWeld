"""Plot the saved reference and measured trajectory without starting Isaac Sim."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    records = np.genfromtxt(args.output_dir / "tcp_execution.csv", delimiter=",", names=True,
                            dtype=None, encoding="utf-8")
    if records.size < 2:
        raise RuntimeError("Need at least two execution samples")
    with np.load(args.output_dir / "trajectory.npz", allow_pickle=False) as trajectory:
        surface = trajectory["surface_points"]
    trace = records[records["phase"] == "trace"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    axes[0].plot(surface[:, 0], surface[:, 1], color="tab:blue", label="Image-derived contour", linewidth=2)
    axes[0].plot(trace["actual_x"], trace["actual_y"], color="tab:orange", linestyle="--", label="Measured TCP")
    axes[0].scatter(*surface[0, :2], color="red", label="Start / closure", zorder=4)
    axes[0].set(xlabel="World X (m)", ylabel="World Y (m)", title="Contour pass: top view", aspect="equal")
    axes[0].legend(fontsize=8)
    elapsed = records["time_s"] - records["time_s"][0]
    clearance = (records["actual_z"] - surface[0, 2]) * 1000
    axes[1].plot(elapsed, clearance, label="Measured TCP clearance")
    axes[1].axhline(10, color="tab:orange", linestyle="--", label="10 mm trace height")
    axes[1].axhline(100, color="tab:green", linestyle=":", label="100 mm approach height")
    axes[1].set(xlabel="Simulation time (s)", ylabel="Height above plate (mm)", title="Approach, trace and retreat")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    target = args.output_dir / "trajectory_validation.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
