"""Load the generated scene into the Isaac Sim 6 full WebRTC experience."""

from pathlib import Path

from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.rendering_manager import ViewportManager


scene_path = Path(__file__).resolve().parent / "outputs" / "quadweld_scene.usda"
if not scene_path.is_file():
    raise FileNotFoundError(f"Generate the scene first: {scene_path}")

opened, stage = stage_utils.open_stage(str(scene_path))
if not opened or stage is None:
    raise RuntimeError(f"Could not open the QuadWeld scene: {scene_path}")
for prim_path in (
    "/World/RobotMount/Robot",
    "/World/RobotStand",
    "/World/Workpiece/SteelPlate",
    "/World/Sensors/OverheadRGB",
):
    if not stage.GetPrimAtPath(prim_path).IsValid():
        raise RuntimeError(f"QuadWeld scene is missing {prim_path}: {scene_path}")

ViewportManager.set_camera_view(
    ViewportManager.get_camera(), eye=[1.45, -1.45, 1.4], target=[0.45, -0.12, 0.45]
)
print(f"QuadWeld streaming scene loaded: {scene_path}", flush=True)
