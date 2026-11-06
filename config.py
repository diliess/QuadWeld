"""Stage-one scene parameters. All distances are in metres."""

from dataclasses import dataclass, replace
import math
from pathlib import Path


@dataclass(frozen=True)
class WorkpieceConfig:
    """Procedural fixtures; local Z=0 is the plate top or inside box floor."""

    kind: str = "outline"
    grid_count: int = 2
    tile_size_m: float = 0.20
    gap_m: float = 0.003
    thickness_m: float = 0.006
    box_size_m: float = 0.40
    wall_height_m: float = 0.12

    def __post_init__(self):
        if self.kind not in ("outline", "plate-grid", "open-box"):
            raise ValueError(f"Unknown workpiece: {self.kind}")
        if not isinstance(self.grid_count, int) or self.grid_count < 2:
            raise ValueError("grid_count must be an integer >= 2")
        dimensions = (self.tile_size_m, self.gap_m, self.thickness_m,
                      self.box_size_m, self.wall_height_m)
        if not all(math.isfinite(value) and value > 0 for value in dimensions):
            raise ValueError("Workpiece dimensions must be finite and positive (metres)")
        if self.box_size_m <= 2 * self.thickness_m:
            raise ValueError("Box size must exceed twice its wall thickness")

    @property
    def grid_extent_m(self) -> float:
        return self.grid_count * self.tile_size_m + (self.grid_count - 1) * self.gap_m


@dataclass(frozen=True)
class SceneConfig:
    # World is Z-up. Robot profiles share the same visual planning parameters.
    robot_name: str = "ur5e"
    robot_mount_prim_path: str = "/World/RobotMount"
    robot_prim_path: str = "/World/RobotMount/Robot"
    robot_usd_relative_path: str = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
    robot_local_usd: str | None = None
    robot_tool_link: str = "wrist_3_link"
    robot_flange_joint: str | None = "joints/robot_gripper_joint"
    flange_translation: tuple[float, float, float] = (0., 0., 0.)
    flange_quaternion_wxyz: tuple[float, float, float, float] = (1., 0., 0., 0.)
    robot_base_position_world: tuple[float, float, float] = (0.45, -0.62, 0.45)
    robot_base_quaternion_wxyz: tuple[float, float, float, float] = (1., 0., 0., 0.)
    robot_stand_size_xy: tuple[float, float] | None = (0.26, 0.26)  # None: the robot stands on the floor.
    robot_base_frame: str = "base_link"
    end_effector_frame: str = "tool0"
    joint_names: tuple[str, ...] = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    tcp_offset_m: float = 0.01  # Clearance above the plate, NOT the tool length.
    tool_length_m: float = 0.12  # Flange to simplified torch tip, along tool +Z.
    tool_marker: bool = True  # Draw the simplified cylinder; off when the asset models the torch.
    home_joint_positions: tuple[float, ...] = (-1.5708, -1.5708, -1.5708, -1.5708, 1.5708, 0.0)
    # Joints outside the arm chain held at fixed targets [rad], e.g. legs of a floating base.
    hold_joint_positions: tuple[tuple[str, float], ...] = ()
    hold_stiffness: float = 0.0
    hold_damping: float = 0.0
    settle_steps: int = 60  # App frames at home before the base frame is read.
    # Feet defining the support plane (contact centres, sphere radius); scales monocular depth.
    foot_links: tuple[str, ...] = ()
    foot_radius_m: float = 0.0
    tool_hull_file: str | None = None  # (N, 3) torch collider vertices in the TCP frame, .npy
    waypoint_spacing_m: float = 0.01
    approach_height_m: float = 0.10
    cartesian_speed_m_s: float = 0.04
    joint_speed_rad_s: float = 0.5
    joint_acceleration_rad_s2: float = 1.0
    physics_dt: float = 1.0 / 60.0
    position_tolerance_m: float = 0.002
    orientation_tolerance_rad: float = 0.02
    waypoint_timeout_s: float = 4.0
    ik_position_tolerance_m: float = 0.0002
    ik_orientation_tolerance_rad: float = 0.002
    max_joint_step_rad: float = 0.35
    drive_stiffness: float = 100000.0
    drive_damping: float = 5000.0
    max_tracking_error_m: float = 0.05
    dark_threshold: int = 80
    min_contour_area_px: float = 300.0

    plate_center_world: tuple[float, float, float] = (0.45, 0.0, 0.5)
    plate_size: tuple[float, float, float] = (0.8, 0.6, 0.01)
    plate_frame_path: str = "/World/Workpiece/PlateFrame"
    rectangle_size: tuple[float, float] = (0.3, 0.2)
    outline_width: float = 0.008
    outline_thickness: float = 0.0004
    outline_clearance: float = 0.0001
    workpiece: WorkpieceConfig = WorkpieceConfig()
    floor_center_world: tuple[float, float, float] = (0.35, -0.15, -0.01)
    floor_size: tuple[float, float, float] = (2.0, 1.9, 0.02)

    camera_prim_path: str = "/World/RobotMount/Sensors/BodyRGBD"
    # Parent the camera to the robot_base_frame rigid body, so it follows a floating base.
    camera_on_base_link: bool = False
    camera_calibration_file: str = "body_rgbd.json"
    camera_resolution: tuple[int, int] = (640, 480)  # Width, height in pixels.
    camera_focal_length_m: float = 0.024
    camera_near_m: float = 0.05
    camera_far_m: float = 10.0

    @property
    def plate_top_z_world(self) -> float:
        return self.plate_center_world[2] + self.plate_size[2] / 2.0


SCENE = SceneConfig()
ROOT = Path(__file__).resolve().parent


def scene_config_for_robot(name: str) -> SceneConfig:
    """Select installation and control parameters without changing visual inference."""
    if name == "ur5e":
        return SCENE
    if name == "rm65":
        return replace(
            SCENE, robot_name="rm65",
            robot_local_usd=str(Path(__file__).resolve().parent /
                                "assets/robots/rm65/RM65-6F/RM65-6F.usda"),
            robot_tool_link="link_6", robot_flange_joint=None,
            joint_names=tuple(f"joint_{i}" for i in range(1, 7)),
            # Smaller reachable demo with the 120 mm tool pointing downward.
            robot_base_position_world=(0.45, -0.38, 0.60),
            robot_stand_size_xy=(0.12, 0.12), rectangle_size=(0.20, 0.10),
            camera_calibration_file="rm65_body_rgbd.json",
            # TCP near (0.20, 0.15, 0.15) in base frame, clear of the camera's task view.
            home_joint_positions=(-2.498098, 0.180659, 1.629718, 0., 1.331208, -2.498095),
            drive_stiffness=10000., drive_damping=500.,
        )
    if name == "b2_e10l":
        return replace(
            SCENE, robot_name="b2_e10l",
            robot_local_usd=str(ROOT / "assets/robots/b2_e10l/B2-E10L/B2-E10L.usda"),
            robot_tool_link="e10l_Link6", robot_flange_joint=None, robot_base_frame="e10l_base_link",
            joint_names=tuple(f"e10l_joint{i}" for i in range(1, 7)),
            # Link6 to the torch wire tip, +Z along the nozzle; see assets/robots/b2_e10l/manifest.json.
            flange_translation=(0.001672, 0.000872, 0.51363),
            flange_quaternion_wxyz=(0.060992, -0.49826, -0.035167, -0.864165),
            tool_length_m=0.0, tool_marker=False,
            # B2 trunk faces the plate (+y), feet just above the floor at the standing pose. With the
            # torch pointing down its wrist sits 0.44 m behind the tip, so the stance keeps the wrist
            # ahead of the B2 head while the arm stays 0.45 rad inside its joint limits.
            robot_base_position_world=(0.45, -1.15, 0.55),
            robot_base_quaternion_wxyz=(math.cos(math.pi / 4), 0., 0., math.sin(math.pi / 4)),
            robot_stand_size_xy=None,
            hold_joint_positions=tuple((f"{leg}_{part}_joint", angle) for leg in ("FL", "FR", "RL", "RR")
                                       for part, angle in (("hip", 0.), ("thigh", 0.75), ("calf", -1.5))),
            hold_stiffness=1800., hold_damping=100.,
            foot_links=tuple(f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")), foot_radius_m=0.032,
            tool_hull_file=str(ROOT / "assets/robots/b2_e10l/torch_hull_tcp.npy"),
            floor_center_world=(0.45, -0.45, -0.01), floor_size=(2.4, 2.6, 0.02),
            camera_on_base_link=True, camera_calibration_file="b2_e10l_body_rgbd.json",
            # Torch raised above and ahead of the camera's view, on the IK branch of the plate approach.
            home_joint_positions=(-0.001377, 0.415495, 1.922079, 0.00214, 0.588969, -0.143391),
            drive_stiffness=400000., drive_damping=10000.,
            # Foot contacts of the 124 kg robot drift at 60 Hz; 200 Hz holds the stance.
            physics_dt=1.0 / 200.0, settle_steps=120,
        )
    raise ValueError(f"Unknown robot profile: {name}")
