"""USD robot frame discovery and task-tool authoring, independent of asset hierarchy."""

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def rigid_body_path(stage, robot_path, name):
    matches = [p.GetPath() for p in Usd.PrimRange(stage.GetPrimAtPath(robot_path))
               if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one rigid body named {name} under {robot_path}; found {matches}")
    return str(matches[0])


def articulation_path(stage, robot_path):
    matches = [p.GetPath() for p in Usd.PrimRange(stage.GetPrimAtPath(robot_path))
               if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one articulation root under {robot_path}; found {matches}")
    return str(matches[0])


def attach_tool(stage, config):
    link_path = rigid_body_path(stage, config.robot_prim_path, config.robot_tool_link)
    flange_path = f"{link_path}/{config.end_effector_frame}"
    position = Gf.Vec3d(*config.flange_translation)
    orientation = Gf.Quatf(*config.flange_quaternion_wxyz)
    if config.robot_flange_joint:
        joint_path = f"{config.robot_prim_path}/{config.robot_flange_joint}"
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(joint_path))
        if not joint or [str(p) for p in joint.GetBody0Rel().GetTargets()] != [link_path]:
            raise RuntimeError(f"Invalid flange mount joint for {config.robot_name}: {joint_path}")
        position = Gf.Vec3d(joint.GetLocalPos0Attr().Get())
        orientation = Gf.Quatf(joint.GetLocalRot0Attr().Get())
        if not joint.GetBody1Rel().GetTargets():
            joint.CreateJointEnabledAttr(False)
    if not stage.GetPrimAtPath(flange_path):
        flange = UsdGeom.Xform.Define(stage, flange_path)
        flange.AddTranslateOp().Set(position)
        flange.AddOrientOp().Set(orientation)
    tool = UsdGeom.Cylinder.Define(stage, f"{flange_path}/Torch")
    tool.CreateRadiusAttr(0.008)
    tool.CreateHeightAttr(config.tool_length_m)
    tool.CreateAxisAttr("Z")
    tool.AddTranslateOp().Set(Gf.Vec3d(0, 0, config.tool_length_m / 2))
    tool.CreateDisplayColorAttr([(0.8, 0.4, 0.1)])
    tcp = UsdGeom.Xform.Define(stage, f"{flange_path}/TCP")
    tcp.AddTranslateOp().Set(Gf.Vec3d(0, 0, config.tool_length_m))
    tip = UsdGeom.Sphere.Define(stage, f"{flange_path}/TCP/Tip")
    tip.CreateRadiusAttr(0.004)
    tip.CreateDisplayColorAttr([(1.0, 0.2, 0.0)])
