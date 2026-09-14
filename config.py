"""Stage-one scene parameters. All distances are in metres."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneConfig:
    # World is Z-up. The UR5e asset has six actuated joints and a fixed base.
    robot_mount_prim_path: str = "/World/RobotMount"
    robot_prim_path: str = "/World/RobotMount/Robot"
    robot_usd_relative_path: str = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
    robot_base_position_world: tuple[float, float, float] = (0.45, -0.62, 0.45)
    robot_stand_size_xy: tuple[float, float] = (0.26, 0.26)
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
    tcp_offset_m: float = 0.01  # Reserved for the later motion stage.

    plate_center_world: tuple[float, float, float] = (0.45, 0.0, 0.5)
    plate_size: tuple[float, float, float] = (0.8, 0.6, 0.01)
    plate_frame_path: str = "/World/Workpiece/PlateFrame"
    rectangle_size: tuple[float, float] = (0.3, 0.2)
    outline_width: float = 0.008
    outline_thickness: float = 0.0004
    outline_clearance: float = 0.0001

    camera_prim_path: str = "/World/Sensors/OverheadRGB"
    camera_position_world: tuple[float, float, float] = (0.45, 0.0, 1.75)
    camera_resolution: tuple[int, int] = (640, 480)  # Width, height in pixels.
    camera_focal_length_m: float = 0.024
    camera_horizontal_aperture_m: float = 0.020955
    camera_near_m: float = 0.05
    camera_far_m: float = 10.0

    @property
    def plate_top_z_world(self) -> float:
        return self.plate_center_world[2] + self.plate_size[2] / 2.0


SCENE = SceneConfig()
