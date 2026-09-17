"""Self-contained USD plate assemblies, with real gaps and individual colliders.

Only scene construction uses these dimensions. No seam coordinates are supplied
to perception or control. Assets contain geometry/materials, not seam overlays.
"""

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from config import WorkpieceConfig


def material(stage, path, color, *, roughness=0.85, metallic=0.0):
    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def cuboid(stage, path, center, size, surface_material, *, collision):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(surface_material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def grid_centers(config: WorkpieceConfig):
    """Fixture placement only; intentionally not a perception input."""
    step = config.tile_size_m + config.gap_m
    for row in range(config.grid_count):
        for col in range(config.grid_count):
            yield row, col, ((col - (config.grid_count - 1) / 2) * step,
                            (row - (config.grid_count - 1) / 2) * step)


def export_workpiece_asset(destination, config: WorkpieceConfig):
    """Write a portable asset with /Workpiece as its default prim, in metres."""
    if config.kind == "outline":
        raise ValueError("The legacy outline is built directly in the main scene")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Workpiece")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("workpieceType", config.kind)
    root.GetPrim().SetDocumentation("Static steel assembly. Origin: top of plates / inside box floor. Units: metres.")
    UsdGeom.Scope.Define(stage, "/Workpiece/Looks")
    UsdGeom.Xform.Define(stage, "/Workpiece/Parts")
    steel = material(stage, "/Workpiece/Looks/Steel", (0.52, 0.56, 0.60),
                     roughness=0.48, metallic=0.7)
    thickness = config.thickness_m
    if config.kind == "plate-grid":
        for row, col, (x, y) in grid_centers(config):
            cuboid(stage, f"/Workpiece/Parts/Plate_{row}_{col}", (x, y, -thickness / 2),
                   (config.tile_size_m, config.tile_size_m, thickness), steel, collision=True)
    else:
        width, height = config.box_size_m, config.wall_height_m
        wall_center = (width - thickness) / 2
        specs = [
            ("Bottom", (0, 0, -thickness / 2), (width, width, thickness)),
            ("LeftWall", (-wall_center, 0, height / 2), (thickness, width, height)),
            ("RightWall", (wall_center, 0, height / 2), (thickness, width, height)),
            ("FrontWall", (0, -wall_center, height / 2), (width - 2 * thickness, thickness, height)),
            ("BackWall", (0, wall_center, height / 2), (width - 2 * thickness, thickness, height)),
        ]
        for name, center, size in specs:
            cuboid(stage, f"/Workpiece/Parts/{name}", center, size, steel, collision=True)
    if not stage.GetRootLayer().Export(str(destination)):
        raise OSError(f"Could not export workpiece: {destination}")
    return destination


def add_workpiece(stage, config, asset_path, stand_material):
    """Reference an exported workpiece and create its supporting fixture."""
    workpiece = config.workpiece
    x, y, _ = config.plate_center_world
    top = config.plate_top_z_world
    frame = UsdGeom.Xform.Define(stage, config.plate_frame_path)
    frame.AddTranslateOp().Set(Gf.Vec3d(x, y, top))
    assembly = UsdGeom.Xform.Define(stage, "/World/Workpiece/Assembly")
    assembly.AddTranslateOp().Set(Gf.Vec3d(x, y, top))
    model = stage.DefinePrim("/World/Workpiece/Assembly/Model", "Xform")
    model.GetReferences().AddReference(str(Path(asset_path).resolve()))

    # Grid pads leave a real air gap below seams; the plinth does not fill them.
    stand_top = top - workpiece.thickness_m
    pad_height = 0.04 if workpiece.kind == "plate-grid" else 0.0
    stand_height = stand_top - pad_height
    extent = workpiece.grid_extent_m if workpiece.kind == "plate-grid" else workpiece.box_size_m
    if stand_height <= 0:
        raise ValueError("Workpiece must be high enough for its support fixture")
    cuboid(stage, "/World/Workpiece/Stand", (x, y, stand_height / 2),
           (extent * 0.9, extent * 0.9, stand_height), stand_material, collision=True)
    if workpiece.kind == "plate-grid":
        UsdGeom.Xform.Define(stage, "/World/Workpiece/Supports")
        pad_size = min(0.05, workpiece.tile_size_m * 0.5)
        for row, col, (dx, dy) in grid_centers(workpiece):
            cuboid(stage, f"/World/Workpiece/Supports/Pad_{row}_{col}",
                   (x + dx, y + dy, stand_height + pad_height / 2),
                   (pad_size, pad_size, pad_height), stand_material, collision=True)
