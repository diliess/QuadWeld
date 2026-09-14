"""Standalone Isaac Sim 6 stage-one scene. No vision or robot motion."""

import argparse
import importlib.metadata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the QuadWeld Isaac Sim 6 scene")
    parser.add_argument("--headless", action="store_true", help="Render once and exit without the GUI")
    parser.add_argument("--robot-usd", help="Local UR5e USD; defaults to the Isaac Sim assets root")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs", help="Saved USD and RGB preview directory"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        isaacsim_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        isaacsim_version = None  # A non-pip Isaac Sim installation can use python.sh.
    if isaacsim_version is not None and not isaacsim_version.startswith("6."):
        raise RuntimeError(f"Isaac Sim 6 is required; this Python has Isaac Sim {isaacsim_version}")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting"})
    try:
        # Kit-dependent modules must be imported after SimulationApp starts.
        from isaacsim.core.experimental.utils import app as app_utils
        from isaacsim.core.experimental.utils import stage as stage_utils
        from isaacsim.core.experimental.prims import Articulation
        from isaacsim.core.rendering_manager import ViewportManager
        from isaacsim.core.simulation_manager import SimulationManager

        from config import SCENE
        from scene.camera import create_rgb_sensor, save_rgb_frame
        from scene.create_scene import create_scene

        stage_utils.create_new_stage()
        camera, robot_usd = create_scene(SCENE, args.robot_usd)
        if not args.headless:
            ViewportManager.set_camera_view(
                ViewportManager.get_camera(), eye=[1.5, -1.45, 1.55], target=[0.3, 0.0, 0.45]
            )

        SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stage_path = args.output_dir / "quadweld_scene.usda"
        if not stage_utils.get_current_stage().GetRootLayer().Export(str(stage_path)):
            raise RuntimeError(f"Could not export scene to {stage_path}")

        sensor = create_rgb_sensor(camera, SCENE)
        robot = Articulation(SCENE.robot_prim_path)
        app_utils.play(commit=True)
        app_utils.update_app(steps=10)
        if robot.num_dofs != 6:
            raise RuntimeError(f"Expected a six-DOF UR5e articulation, found {robot.num_dofs} DOFs")
        preview_path = args.output_dir / "overhead_rgb.png"
        save_rgb_frame(sensor, simulation_app, preview_path)
        print(f"Robot USD: {robot_usd}")
        print(f"Robot DOFs: {robot.num_dofs}")
        print(f"Scene saved: {stage_path}")
        print(f"RGB image saved: {preview_path}")

        if not args.headless:
            while simulation_app.is_running():
                simulation_app.update()
        app_utils.stop(commit=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
