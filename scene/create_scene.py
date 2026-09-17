"""Create the robot, selected workpiece, lighting, and body RGB-D camera."""

from pathlib import Path

from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom, UsdLux

from config import SceneConfig
from scene.camera import create_body_camera
from scene.workpieces import add_workpiece, cuboid as _cuboid, material as _material
from robot.asset import attach_tool


def _resolve_robot_usd(config: SceneConfig, robot_usd_override: str | None) -> str:
    if robot_usd_override or config.robot_local_usd:
        robot_usd = Path(robot_usd_override or config.robot_local_usd).expanduser().resolve()
        if not robot_usd.is_file():
            raise FileNotFoundError(f"Robot USD does not exist: {robot_usd}")
        return str(robot_usd)

    try:
        assets_root = get_assets_root_path()
    except RuntimeError as exc:
        raise RuntimeError(
            "Isaac Sim assets are unavailable. Configure the official assets root "
            "or pass --robot-usd /absolute/path/to/ur5e.usd."
        ) from exc
    if not assets_root:
        raise RuntimeError(
            "Isaac Sim assets root was not found. Configure the official assets root "
            "or pass --robot-usd /absolute/path/to/ur5e.usd."
        )
    return assets_root.rstrip("/") + config.robot_usd_relative_path


def create_scene(config: SceneConfig, robot_usd_override: str | None = None, *, camera_calibration,
                 open_outline: bool = False, workpiece_asset: Path | None = None):
    stage = stage_utils.get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world_prim)

    UsdGeom.Xform.Define(stage, "/World/Lights")
    light = UsdLux.DomeLight.Define(stage, "/World/Lights/Ambient")
    light.CreateIntensityAttr(600.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    floor_material = _material(stage, "/World/Looks/Floor", (0.20, 0.23, 0.27))
    stand_material = _material(stage, "/World/Looks/Stand", (0.14, 0.16, 0.19))
    plate_material = _material(stage, "/World/Looks/SteelPlate", (0.72, 0.76, 0.79))
    line_material = _material(stage, "/World/Looks/Rectangle", (0.008, 0.008, 0.008))

    _cuboid(stage, "/World/Floor", (0.35, -0.15, -0.01), (2.0, 1.9, 0.02), floor_material, collision=True)

    base_x, base_y, base_z = config.robot_base_position_world
    if base_z > 0.0:
        _cuboid(
            stage,
            "/World/RobotStand",
            (base_x, base_y, base_z / 2.0),
            (*config.robot_stand_size_xy, base_z),
            stand_material,
            collision=True,
        )

    robot_mount = UsdGeom.Xform.Define(stage, config.robot_mount_prim_path)
    robot_mount.AddTranslateOp().Set(Gf.Vec3d(*config.robot_base_position_world))
    robot_usd = _resolve_robot_usd(config, robot_usd_override)
    stage_utils.add_reference_to_stage(usd_path=robot_usd, path=config.robot_prim_path)
    attach_tool(stage, config)

    UsdGeom.Xform.Define(stage, "/World/Workpiece")
    if config.workpiece.kind == "outline":
        _create_outline(stage, config, stand_material, plate_material, line_material, open_outline)
    else:
        if workpiece_asset is None:
            raise ValueError("A generated USD asset is required for this workpiece")
        add_workpiece(stage, config, workpiece_asset, stand_material)
        # A broad angled light makes the physical edges and inside corners visible.
        key = UsdLux.RectLight.Define(stage, "/World/Lights/WorkpieceKey")
        key.CreateIntensityAttr(1600.0)
        key.CreateWidthAttr(0.65)
        key.CreateHeightAttr(0.65)
        key.AddTranslateOp().Set(Gf.Vec3d(0.05, -0.40, 1.40))
        key.AddRotateXYZOp().Set(Gf.Vec3f(20.0, -25.0, 0.0))

    UsdGeom.Xform.Define(stage, f"{config.robot_mount_prim_path}/Sensors")
    camera = create_body_camera(config, camera_calibration)
    return camera, robot_usd


def _create_outline(stage, config, stand_material, plate_material, line_material, open_outline):
    plate_x, plate_y, plate_z = config.plate_center_world
    plate_thickness = config.plate_size[2]
    plate_frame = UsdGeom.Xform.Define(stage, config.plate_frame_path)
    UsdGeom.XformCommonAPI(plate_frame).SetTranslate(Gf.Vec3d(plate_x, plate_y, config.plate_top_z_world))

    # The stand top touches the bottom of the steel plate.
    stand_height = plate_z - plate_thickness / 2.0
    _cuboid(
        stage,
        "/World/Workpiece/Stand",
        (plate_x, plate_y, stand_height / 2.0),
        (0.42, 0.36, stand_height),
        stand_material,
        collision=True,
    )
    _cuboid(stage, "/World/Workpiece/SteelPlate", config.plate_center_world, config.plate_size, plate_material, collision=True)

    UsdGeom.Xform.Define(stage, "/World/Workpiece/Rectangle")
    rectangle_length, rectangle_width = config.rectangle_size
    line_width = config.outline_width
    line_z = config.plate_top_z_world + config.outline_clearance + config.outline_thickness / 2.0
    outline_specs = (
        ("Top", (plate_x, plate_y + rectangle_width / 2.0, line_z), (rectangle_length, line_width, config.outline_thickness)),
        ("Bottom", (plate_x, plate_y - rectangle_width / 2.0, line_z), (rectangle_length, line_width, config.outline_thickness)),
        ("Left", (plate_x - rectangle_length / 2.0, plate_y, line_z), (line_width, rectangle_width - line_width, config.outline_thickness)),
        ("Right", (plate_x + rectangle_length / 2.0, plate_y, line_z), (line_width, rectangle_width - line_width, config.outline_thickness)),
    )
    for side, center_world, size in outline_specs:
        if open_outline and side == "Top":
            continue  # The three remaining sides form the sample endpoint-to-endpoint U path.
        _cuboid(
            stage,
            f"/World/Workpiece/Rectangle/{side}",
            center_world,
            size,
            line_material,
            collision=False,
        )
