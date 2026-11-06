"""Reference portable scene assets (metre/Z-up USD with a default prim)."""

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom


def reference_asset(stage, asset_path, prim_path, *, at=(0., 0., 0.)):
    """Reference an asset's default prim at a parent-frame translation; retain its local origin."""
    asset_path = Path(asset_path).expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Asset USD does not exist: {asset_path}")
    source = Usd.Stage.Open(str(asset_path))
    if not source or not source.GetDefaultPrim():
        raise ValueError(f"Asset USD needs a default prim: {asset_path}")
    if UsdGeom.GetStageUpAxis(source) != UsdGeom.Tokens.z:
        raise ValueError(f"Asset USD must use Z-up: {asset_path}")
    if abs(UsdGeom.GetStageMetersPerUnit(source) - 1.) > 1e-9:
        raise ValueError(f"Asset USD must use metres: {asset_path}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.) > 1e-9:
        raise ValueError("Destination stage must use metres and Z-up")
    path = Sdf.Path(prim_path)
    if not path.IsAbsoluteRootOrPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError("Asset prim_path must be an absolute prim path")
    if stage.GetPrimAtPath(path):
        raise ValueError(f"Asset prim already exists: {path}")
    frame = UsdGeom.Xform.Define(stage, path)
    frame.GetPrim().GetReferences().AddReference(str(asset_path))
    frame.AddTranslateOp().Set(Gf.Vec3d(*at))
    return frame.GetPrim()
