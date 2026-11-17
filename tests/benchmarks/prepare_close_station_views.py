"""Plan rated-range, arm-mounted, unobstructed T-joint views before inference."""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tests.benchmarks.seam_metrics import digest,write_json
from weld_scene.overview_planning import static_check


def solve_reach(planner,pose,yaw,reach):
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()
    base[2,3]=planner.source_base[2,3]
    target=base[:3,:3].T@pose[:3,:3]
    def residual(q,regularize=False):
        camera=planner.inverse_base@planner.arm.camera(q)
        error=np.r_[camera[2,3]-(pose[2,3]-base[2,3]),camera[0,3]-reach,
                    .35*Rotation.from_matrix(target@camera[:3,:3].T).as_rotvec()]
        return np.r_[error,.004*(q-planner.arm.initial)] if regularize else error
    fit=least_squares(lambda q:residual(q,True),planner.arm.initial,bounds=planner.arm.bounds.T,max_nfev=100)
    exact=least_squares(residual,fit.x,bounds=planner.arm.bounds.T,max_nfev=180,gtol=1e-9,xtol=1e-9,ftol=1e-9)
    if np.linalg.norm(residual(exact.x))>2e-5:return None
    camera=planner.inverse_base@planner.arm.camera(exact.x)
    base[:2,3]=pose[:2,3]-(base[:3,:3]@camera[:3,3])[:2]
    return base,exact.x


def prepare(args):
    import numpy as np
    from pxr import Gf,Sdf,Usd,UsdGeom
    from weld_scene.arm_kinematics import ArmKinematics
    from weld_scene.mount_offset import apply_usd_offset,apply_collision_offset,torch_frustum_audit
    from weld_scene.optics import load_calibration
    from weld_scene.overview_planning import OverviewPlanner,t_geometry,transform_points,world_matrix,look_at,framing

    previous=json.loads((args.baseline_run/'station_plan.json').read_text())
    source=Path(previous['source_usd'])
    stage=Usd.Stage.Open(str(source));stage.SetEditTarget(stage.GetSessionLayer())
    mount=apply_usd_offset(stage,previous['mount_adjustment']['lateral_offset_m'])
    calibration=load_calibration()
    planner=OverviewPlanner(stage,previous['collision_model'],calibration)
    apply_collision_offset(planner.model,mount)
    records=[]
    for old in previous['views']:
        station=old['station'];fixture,bounds,_,_=t_geometry(stage,station)
        lo,hi=bounds['web'][:,0];centres=np.linspace(lo+(hi-lo)/6,hi-(hi-lo)/6,3)
        selected=None;attempts=[]
        for elevation in (35,25,45,15):
            if selected:break
            for reach in (.60,.45,.75):
                if selected:break
                for yaw_delta in (0,-25,25):
                    poses=[];passed=True
                    for label,x in zip(('left','center','right'),centres):
                        target=np.array([x,bounds['web'][0,1],bounds['base'][1,2]])
                        angle=np.radians(elevation)
                        eye=target+args.distance_m*np.array([0.,-np.cos(angle),np.sin(angle)])
                        pose=look_at(transform_points(fixture,eye[None])[0],transform_points(fixture,target[None])[0])
                        yaw=np.arctan2(pose[1,2],pose[0,2])+np.radians(yaw_delta)
                        solution=solve_reach(planner,pose,yaw,reach)
                        if solution is None:passed=False;break
                        base,q=solution;valid,check=static_check(planner,base,q)
                        if not valid:passed=False;break
                        # Physical 300 mm seam patch, using CAD contact geometry only.
                        patch=np.array([[x-.15,target[1],target[2]],[x+.15,target[1],target[2]]])
                        fraction,_=framing(pose,transform_points(fixture,patch),calibration['intrinsics'],calibration['image_size_wh'])
                        if fraction<1:passed=False;break
                        poses.append({'id':station+'_near_'+label,'station':station,'patch':label,
                            'fixture_patch_center_x_m':float(x),'target_distance_m':args.distance_m,
                            'world_from_optical_midpoint':pose.tolist(),'base_world':base.tolist(),
                            'arm_q_rad':q.tolist(),'arm_q_deg':np.degrees(q).tolist(),
                            'elevation_deg':elevation,'base_yaw_delta_deg':yaw_delta,'arm_reach_x_m':reach,
                            'cad_patch_in_both_eyes_with_margin':True,**check})
                    attempts.append({'elevation_deg':elevation,'reach_m':reach,'base_yaw_delta_deg':yaw_delta,'all_three_passed':passed})
                    if passed:selected=poses;break
        if selected is None:raise ValueError('No close pose for '+station+': '+str(attempts))
        write_json(args.output_dir/(station+'_pose_attempts.json'),attempts)
        for view in selected:
            folder=args.output_dir/'captures'/view['id'];folder.mkdir(parents=True)
            layer=Sdf.Layer.CreateNew(str(folder/'arm_pose.usda'));layer.subLayerPaths=[old['usd']]
            placed=Usd.Stage.Open(layer)
            UsdGeom.SetStageMetersPerUnit(placed,1.);UsdGeom.SetStageUpAxis(placed,'Z')
            placed.SetDefaultPrim(placed.GetPrimAtPath('/World'))
            cache=UsdGeom.XformCache()
            current_base=world_matrix(cache,placed.GetPrimAtPath('/World/Robot/base_link'))
            current_robot=world_matrix(cache,placed.GetPrimAtPath('/World/Robot'))
            delta=np.array(view['base_world'])@np.linalg.inv(current_base)
            UsdGeom.Xformable(placed.GetPrimAtPath('/World/Robot')).MakeMatrixXform().Set(Gf.Matrix4d((delta@current_robot).T.tolist()))
            arm=ArmKinematics(placed);arm.apply(view['arm_q_rad'])
            np.testing.assert_allclose(arm.camera(view['arm_q_rad']),view['world_from_optical_midpoint'],atol=2e-5)
            camera_paths=old['camera_paths'];view.update(usd=str(folder/'arm_pose.usda'),camera_paths=camera_paths,
                context_camera=old['context_camera'])
            layer.Save();records.append(view)
        print('CLOSE_STATION_PLANNED',station,'distance',args.distance_m,'elevation',selected[0]['elevation_deg'],
              'reach',selected[0]['arm_reach_x_m'],'q_deg',selected[0]['arm_q_deg'],flush=True)
    audit=torch_frustum_audit(stage,calibration)
    if any(r['triangles_potentially_intersecting_frustum'] for r in audit):raise ValueError('Torch still visible')
    plan={**previous,'views':records,'capture_scope':'rated_close_patches','baseline_run':str(args.baseline_run),
        'mount_adjustment':previous['mount_adjustment'],'torch_frustum_audit':audit,'require_no_torch_pixels':True,
        'target_distance_m':args.distance_m,'intended_window_centres_fixture_x_m':centres.tolist(),
        'view_selection':'Fixed 650 mm distance, three overlapping CAD patches per T; first feasible static arm/base poses; no model feedback',
        'source_hashes':{name:digest(ROOT/name) for name in ('tests/benchmarks/prepare_close_station_views.py',
            'weld_scene/overview_planning.py','weld_scene/mount_offset.py','weld_scene/arm_kinematics.py','tests/benchmarks/capture_station_overviews.py')},
        'physics_executed':False,'scope':'Static close work placements; unchanged previously shifted mount; no motion executed'}
    write_json(args.output_dir/'station_plan.json',plan)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run',type=Path,default=ROOT/'outputs/archive/non_stereo_experiments_20261117/outputs/benchmarks/t_station_unobstructed_20261110_01')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--distance-m',type=float,default=.65)
    a=p.parse_args();a.baseline_run=a.baseline_run.resolve();a.output_dir=a.output_dir.resolve()
    if not .4<=a.distance_m<=.75:p.error('Use 0.40–0.75 m to keep working margins inside rated range')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):p.error('Use an empty output directory')
    a.output_dir.mkdir(parents=True,exist_ok=True);prepare(a)
