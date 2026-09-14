"""Isaac Sim 6: image-derived contour, Cartesian path and one complete robot pass."""

import argparse
import importlib.metadata
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the QuadWeld Isaac Sim 6 scene")
    parser.add_argument("--headless", action="store_true", help="Complete the task and exit without the GUI")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scene-only", action="store_true", help="Only create the scene and capture RGB-D")
    mode.add_argument("--plan-only", action="store_true", help="Detect and preflight the path without executing it")
    parser.add_argument("--gpu", type=int, default=0, help="Single RTX GPU index")
    parser.add_argument("--webrtc", action="store_true", help="Stream the live task to a WebRTC client")
    parser.add_argument("--public-ip", help="Server IP reachable by the WebRTC client")
    parser.add_argument("--signal-port", type=int, default=49100, help="WebRTC TCP signaling port")
    parser.add_argument("--stream-port", type=int, default=47998, help="WebRTC UDP media port")
    parser.add_argument("--auto-start", action="store_true", help="Skip the WebRTC Start task gate")
    parser.add_argument("--exit-on-complete", action="store_true", help="Close the application after task completion")
    parser.add_argument("--robot-usd", help="Local UR5e USD; defaults to the Isaac Sim assets root")
    parser.add_argument("--calibration", type=Path,
                        default=Path(__file__).resolve().parent / "calibration/body_rgbd.json",
                        help="RGB-D intrinsics and camera-to-base calibration, independent of workpiece pose")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs", help="Saved USD and RGB preview directory"
    )
    args = parser.parse_args()
    if not all(1 <= port <= 65535 for port in (args.signal_port, args.stream_port)):
        parser.error("Streaming ports must be between 1 and 65535")
    return args


def main() -> None:
    args = parse_args()
    try:
        isaacsim_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        isaacsim_version = None  # A non-pip Isaac Sim installation can use python.sh.
    if isaacsim_version is not None and not isaacsim_version.startswith("6."):
        raise RuntimeError(f"Isaac Sim 6 is required; this Python has Isaac Sim {isaacsim_version}")

    from isaacsim import SimulationApp

    launch_config = {"headless": args.headless or args.webrtc, "renderer": "RaytracedLighting",
                     "active_gpu": args.gpu, "multi_gpu": False}
    if args.webrtc:
        launch_config["hide_ui"] = False
        prefix = "--/exts/omni.kit.livestream.app/primaryStream/"
        launch_config["extra_args"] = [
            prefix + f"signalPort={args.signal_port}", prefix + f"streamPort={args.stream_port}",
            prefix + "enableEventTracing=false",
            "--/exts/omni.services.livestream.session/quitOnSessionEnded=false",
        ]
        if args.public_ip:
            launch_config["extra_args"].append(prefix + f"publicIp={args.public_ip}")
    simulation_app = SimulationApp(launch_config)
    failed = False
    try:
        if args.webrtc:
            import omni.kit.app

            # Enable streaming on the same Python experience used by the controller.
            # The full editor experience has a separate startup/layout lifecycle.
            extensions = omni.kit.app.get_app().get_extension_manager()
            extensions.set_extension_enabled_immediate("omni.kit.livestream.app", True)
            if not extensions.is_extension_enabled("omni.kit.livestream.app"):
                raise RuntimeError("Could not enable Isaac Sim WebRTC streaming")
        # Kit-dependent modules must be imported after SimulationApp starts.
        from isaacsim.core.experimental.utils import app as app_utils
        from isaacsim.core.experimental.utils import stage as stage_utils
        from isaacsim.core.experimental.prims import Articulation
        from isaacsim.core.rendering_manager import ViewportManager
        from isaacsim.core.simulation_manager import SimulationManager

        from config import SCENE
        from scene.camera import create_rgbd_sensor, save_rgbd_frame
        from scene.create_scene import create_scene
        from vision.rgbd import RGBDCalibration, RGBDSettings, generate_rgbd_trajectory, save_rgbd_trajectory

        calibration = RGBDCalibration.load(args.calibration)
        stage_utils.create_new_stage()
        camera, robot_usd = create_scene(SCENE, args.robot_usd, camera_calibration=calibration)
        if not args.headless or args.webrtc:
            ViewportManager.set_camera_view(
                ViewportManager.get_camera(), eye=[1.5, -1.45, 1.55], target=[0.3, 0.0, 0.45]
            )

        SimulationManager.setup_simulation(dt=SCENE.physics_dt, device="cpu")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "rgbd_calibration.json").write_text(args.calibration.read_text())
        # Invalidate a previous run's success before any new planning or execution.
        (args.output_dir / "execution_report.json").write_text(json.dumps(
            {"completed": False, "status": "scene_only" if args.scene_only else "not_executed"}, indent=2) + "\n")
        stage_path = args.output_dir / "quadweld_scene.usda"
        if not stage_utils.get_current_stage().GetRootLayer().Export(str(stage_path)):
            raise RuntimeError(f"Could not export scene to {stage_path}")

        sensor = create_rgbd_sensor(camera, SCENE)
        robot = Articulation(SCENE.robot_prim_path)
        if not args.scene_only:
            import numpy as np
            from robot.kinematics import SerialKinematics
            from robot.robot_controller import RobotController, plan_joint_path
            from scene.debug import draw_plan, draw_polyline

            stage = stage_utils.get_current_stage()
            model = SerialKinematics.from_stage(stage, SCENE)
            world_from_base = model.world_from_base.copy()  # Simulator display/readback adapter only.
            model.world_from_base = np.eye(4)
            model.save(args.output_dir / "robot_kinematics.npz")
        panel = None
        if args.webrtc:
            from scene.streaming import StreamingPanel

            panel = StreamingPanel()
            print(f"WebRTC: TCP {args.signal_port}, UDP {args.stream_port}; "
                  f"server={args.public_ip or 'server network IP'}", flush=True)
            if args.auto_start:
                panel.start()
            else:
                panel.wait_for_start(simulation_app)
            panel.set_status("Initializing robot and capturing RGB-D...")
        app_utils.play(commit=True)
        app_utils.update_app(steps=10)
        if robot.num_dofs != 6:
            raise RuntimeError(f"Expected a six-DOF UR5e articulation, found {robot.num_dofs} DOFs")
        if not args.scene_only:
            controller = RobotController(robot, model, simulation_app, SCENE, args.output_dir,
                                         world_from_base=world_from_base)
            if panel:
                controller.progress_callback = panel.set_status
            controller.initialize_home()
        preview_path = args.output_dir / "body_rgb.png"
        image_rgb, depth_m = save_rgbd_frame(sensor, simulation_app, args.output_dir)
        print(f"Robot USD: {robot_usd}")
        print(f"Robot DOFs: {robot.num_dofs}")
        print(f"Scene saved: {stage_path}")
        print(f"RGB image saved: {preview_path}")

        if not args.scene_only:
            if panel:
                panel.set_status("Detecting contour and checking trajectory...")
            settings = RGBDSettings(waypoint_spacing_m=SCENE.waypoint_spacing_m, tcp_offset_m=SCENE.tcp_offset_m,
                                    approach_height_m=SCENE.approach_height_m, dark_threshold=SCENE.dark_threshold,
                                    min_contour_area_px=SCENE.min_contour_area_px)
            result = generate_rgbd_trajectory(image_rgb, depth_m, calibration, settings)
            save_rgbd_trajectory(result, image_rgb, args.output_dir)
            path = result.path
            controller.observed_surface = result.surface
            joints, poses, phases = plan_joint_path(model, path, controller.current_q(), SCENE,
                                                    update_callback=simulation_app.update if args.webrtc else None,
                                                    surface=result.surface)
            np.savez(args.output_dir / "trajectory.npz", contour_pixel=result.contour_pixel,
                     contour_base=result.contour_base, surface_points=path.surface_points,
                     poses_base=poses, joint_positions=joints, phases=phases,
                     surface_normal_base=result.surface.normal, surface_point_base=result.surface.point,
                     coordinate_frame="robot_base", base_from_camera=calibration.base_from_camera)
            draw_plan(stage, path, world_from_base)
            if not stage.GetRootLayer().Export(str(stage_path)):
                raise RuntimeError(f"Could not export planned scene to {stage_path}")
            print(f"Preflight passed: {len(path.surface_points)} contour samples; {len(joints) - 1} motion targets", flush=True)
            if not args.plan_only:
                controller.follow_cartesian_path(joints, poses, phases)
                actual_world = np.array([row[2:5] for row in controller.history]) @ world_from_base[:3, :3].T + world_from_base[:3, 3]
                draw_polyline(stage, "TCPActual", actual_world, (1.0, 0.4, 0.0))
                if not stage.GetRootLayer().Export(str(args.output_dir / "quadweld_result.usda")):
                    raise RuntimeError("Could not export result scene")
                print("Contour pass complete; TCP lifted to approach height.", flush=True)

        if panel:
            panel.set_status("Task complete. Close the application to exit." if not args.plan_only and not args.scene_only
                             else "Preview complete. Close the application to exit.")

        if (not args.headless or args.webrtc) and not args.exit_on_complete:
            while simulation_app.is_running():
                simulation_app.update()
        app_utils.stop(commit=True)
    except Exception:
        # Kit fast shutdown may terminate before Python renders an uncaught exception.
        import traceback
        failed = True
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(exit_code=int(failed))


if __name__ == "__main__":
    main()
