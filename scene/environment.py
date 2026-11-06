"""Compose a portable environment without importing or selecting a robot."""

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from scene.assets import reference_asset


def add_environment(stage, asset_path, *, prim_path="/World/Environment", at=(0., 0., 0.)):
    """Reference a metre/Z-up USD at a world translation; retain its local origin.

    This adds geometry/materials/physics only. It does not create a physics scene,
    change a robot profile, or provide seam coordinates to perception.
    """
    return reference_asset(stage, asset_path, prim_path, at=at)


def environment_camera_view(stage, prim_path="/World/Environment"):
    """Fit colliders, so a wide visual surround does not shrink the useful view."""
    root = stage.GetPrimAtPath(prim_path)
    if not root:
        raise ValueError(f"Environment not found: {prim_path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], False, True)
    bounds = Gf.Range3d()
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            bounds.UnionWith(cache.ComputeWorldBound(prim).ComputeAlignedRange())
    if bounds.IsEmpty():
        bounds = cache.ComputeWorldBound(root).ComputeAlignedRange()
    centre = bounds.GetMidpoint()
    span = max(*bounds.GetSize(), 1.)
    eye = centre + Gf.Vec3d(1.1 * span, -1.35 * span, 1.1 * span)
    return list(eye), list(centre)
