"""Keep the previous optical views; extend the simulated camera bracket 50 mm."""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark_stereo_seams import digest,write_json


def prepare(args):
    import mujoco as mj
    import numpy as np
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdSemantics
    from scipy.spatial import ConvexHull
    from hj90.arm_kinematics import ArmKinematics
    from hj90.mount_offset import TOOL,apply_collision_offset,apply_usd_offset,torch_frustum_audit
    from hj90.optics import load_calibration
    from hj90.overview_planning import OverviewPlanner,t_geometry,transform_points,world_matrix,framing,look_at

    baseline=json.loads((args.baseline_run/'station_plan.json').read_text())
    source=Path(baseline['source_usd'])
    stage=Usd.Stage.Open(str(source));stage.SetEditTarget(stage.GetSessionLayer())
    mount=apply_usd_offset(stage,args.offset_mm/1000)
    calibration=load_calibration()
    audit=torch_frustum_audit(stage,calibration)
    if any(a['triangles_potentially_intersecting_frustum'] for a in audit):
        raise ValueError('Torch still intersects a stereo frustum: '+str(audit))
    planner=OverviewPlanner(Usd.Stage.Open(str(source)),baseline['collision_model'],calibration)
    apply_collision_offset(planner.model,mount)
    records=[]
    for original in baseline['views']:
        view=copy.deepcopy(original);camera=np.array(view['world_from_optical_midpoint'])
        base=np.array(view['base_world'])
        # Counter-translate the whole robot to preserve the original optical pose.
        base[:3,3] -= camera[:3,0]*args.offset_mm/1000
        planner.set_collision_pose(base,view['arm_q_rad'])
        bad=[float(c.dist) for c in planner.data.contact if c.dist < -.0005]
        feet=planner.data.geom_xpos[planner.feet]
        floor_error=float(np.max(np.abs(feet[:,2]-planner.model.geom_size[planner.feet,0])))
        hull=ConvexHull(feet[:,:2]);com=planner.data.subtree_com[planner.body,:2]
        support_margin=-float(np.max(hull.equations[:,:2]@com+hull.equations[:,2]))
        if bad or floor_error > .001 or support_margin < 0:
            raise ValueError('Static pose audit failed: '+view['station'])
        folder=args.output_dir/'captures'/view['id'];folder.mkdir(parents=True)
        layer=Sdf.Layer.CreateNew(str(folder/'arm_pose.usda'));layer.subLayerPaths=[original['usd']]
        placed=Usd.Stage.Open(layer)
        # Stage units/up axis are root-layer metadata, not inherited from sublayers.
        UsdGeom.SetStageMetersPerUnit(placed,1.)
        UsdGeom.SetStageUpAxis(placed,'Z')
        placed.SetDefaultPrim(placed.GetPrimAtPath('/World'))
        current=np.array(UsdGeom.Xformable(placed.GetPrimAtPath('/World/Robot')).GetLocalTransformation()).T
        current[:3,3] -= camera[:3,0]*args.offset_mm/1000
        UsdGeom.Xformable(placed.GetPrimAtPath('/World/Robot')).MakeMatrixXform().Set(Gf.Matrix4d(current.T.tolist()))
        apply_usd_offset(placed,args.offset_mm/1000)
        # The previously attached rig and the housing are translated together.
        arm=ArmKinematics(placed)
        np.testing.assert_allclose(arm.camera(arm.initial),camera,atol=2e-5)
        cache=UsdGeom.XformCache()
        fixture,_,vertices,surfaces=t_geometry(placed,view['station'])
        vertices,surfaces=transform_points(fixture,vertices),transform_points(fixture,surfaces)
        fraction,boxes=framing(arm.camera(arm.initial),vertices,calibration['intrinsics'],calibration['image_size_wh'])
        if fraction != 1:raise ValueError('Full workpiece framing changed')
        visibility=[]
        for sign in (-1,1):
            eye=camera[:3,3]+sign*.045*camera[:3,0];count=0
            for point in surfaces:
                ray=point-eye;distance=np.linalg.norm(ray);ray/=distance
                hit=mj.mj_ray(planner.model,planner.data,eye,ray,None,1,-1,np.array([-1],np.int32))
                count += hit >= 0 and abs(hit-distance)<.003
            visibility.append(count/len(surfaces))
        for prim in placed.GetPrimAtPath(TOOL).GetChildren():
            if prim.GetName().startswith('e05l_') and prim.IsA(UsdGeom.Mesh):
                UsdSemantics.LabelsAPI.Apply(prim,'class').CreateLabelsAttr(['welding_torch'])
        # Show the entire robot and workbench in the external comparison photo.
        context=UsdGeom.Camera(placed.GetPrimAtPath(view['context_camera']))
        eye=transform_points(fixture,np.array([[3.8,-5.5,3.4]]))[0]
        target=transform_points(fixture,np.array([[0.,-1.,.75]]))[0]
        transform=look_at(eye,target)@np.diag([1.,-1.,-1.,1.])
        context.MakeMatrixXform().Set(Gf.Matrix4d(transform.T.tolist()))
        layer.Save()
        view.update(usd=str(folder/'arm_pose.usda'),base_world=base.tolist(),
            front_surface_ray_visibility_both_eyes=visibility,mesh_normalized_bounds_both_eyes=boxes,
            foot_floor_max_error_m=floor_error,com_support_margin_m=support_margin,
            source_optical_pose_preserved=True)
        view.pop('score',None)
        records.append(view)
        print('UNOBSTRUCTED_PLANNED',view['station'],'CAD visibility',visibility,flush=True)
    plan={**baseline,'views':records,'baseline_run':str(args.baseline_run),'mount_adjustment':mount,
          'torch_frustum_audit':audit,'require_no_torch_pixels':True,
          'view_selection':'Same optical poses, K, baseline and arm q as baseline; physical camera bracket +50 mm, robot base counter-translated',
          'source_hashes':{name:digest(ROOT/name) for name in ('scripts/prepare_unobstructed_overviews.py',
              'hj90/mount_offset.py','scripts/capture_station_overviews.py','hj90/arm_kinematics.py')},
          'physics_executed':False,'scope':'Static synthetic comparison; modified simulated bracket, not a real calibrated mount; bracket mass unknown'}
    write_json(args.output_dir/'station_plan.json',plan)
    print('UNOBSTRUCTED_PLAN_COMPLETE',args.output_dir,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run',type=Path,default=ROOT/'outputs/benchmarks/t_station_overviews_20261110_01')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--offset-mm',type=float,default=50.)
    a=p.parse_args();a.baseline_run=a.baseline_run.resolve();a.output_dir=a.output_dir.resolve()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):p.error('Use an empty output directory')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    prepare(a)
