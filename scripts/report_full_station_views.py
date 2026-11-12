"""Validate captured whole-workpiece stereo framing and show the saved robot poses."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Usd, UsdGeom

from hj90.optics import load_calibration
from hj90.overview_planning import t_geometry, transform_points


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(folder):
    plan = json.loads((folder / 'station_plan.json').read_text())
    status = json.loads((folder / 'capture_status.json').read_text())
    assert status == {'completed': True, 'stereo_pairs': len(plan['views'])}
    assert digest(ROOT / 'assets/scenes/industrial-welding/industrial_welding.usdc') == plan['source_usdc_sha256']
    calibration = load_calibration()
    images = folder / 'images'
    images.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    preview = Image.new('RGB', (1488, 804), '#f2f4f6')
    draw = ImageDraw.Draw(preview)
    rows = []
    for i, view in enumerate(plan['views']):
        capture_dir = folder / 'captures' / view['id']
        capture = json.loads((capture_dir / 'evaluation/capture.json').read_text())
        pair = json.loads((capture_dir / 'input.json').read_text())
        stage = Usd.Stage.Open(str(capture_dir / 'arm_pose.usda'))
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.
        assert UsdGeom.GetStageUpAxis(stage) == 'Z'
        fixture, bounds, vertices, _ = t_geometry(stage, view['station'])
        world_vertices = transform_points(fixture, vertices)
        world_cameras = np.asarray(capture['world_from_camera_optical'])
        np.testing.assert_allclose(world_cameras[0, :3, :3], world_cameras[1, :3, :3], atol=1e-8)
        baseline = np.linalg.norm(world_cameras[0, :3, 3] - world_cameras[1, :3, 3])
        np.testing.assert_allclose(baseline, calibration['baseline_m'], atol=1e-8)
        np.testing.assert_allclose(pair['baseline_m'], baseline, atol=1e-8)
        midpoint = world_cameras[0].copy()
        midpoint[:3, 3] = world_cameras[:, :3, 3].mean(0)
        pose_error = float(np.max(np.abs(midpoint - view['world_from_optical_midpoint'])))
        assert pose_error < 2e-6
        np.testing.assert_allclose(capture['actual_arm_q_rad'], view['arm_q_rad'], atol=1e-6)
        target = np.array([0., 0., bounds['base'][1, 2] + .5 * np.ptp(bounds['web'][:, 2])])
        distance = float(np.linalg.norm(midpoint[:3, 3] - transform_points(fixture, target[None])[0]))
        np.testing.assert_allclose(distance, view['camera_to_workpiece_center_m'], atol=2e-6)
        base_local = transform_points(np.linalg.inv(fixture), np.asarray(capture['base_world'])[:3, 3][None])[0]
        gap = float(np.linalg.norm(base_local[:2] - np.clip(base_local[:2], bounds['base'][0, :2], bounds['base'][1, :2])))
        np.testing.assert_allclose(gap, view['robot_base_center_to_workpiece_xy_m'], atol=1e-8)
        margin = plan['framing_margin_fraction']
        eyes = []
        stereo = Image.new('RGB', (1296, 1148), '#f2f4f6')
        stereo_draw = ImageDraw.Draw(stereo)
        for eye_index, input_view in enumerate(pair['views']):
            side = input_view['id']
            camera = world_cameras[eye_index]
            points = transform_points(np.linalg.inv(camera), world_vertices)
            k = np.asarray(input_view['intrinsics'])
            np.testing.assert_allclose(k, calibration['intrinsics'], atol=1e-3)
            projected = points @ k.T
            normalized = projected[:, :2] / projected[:, 2:] / input_view['image_size_wh']
            assert (points[:, 2] > 0).all()
            assert (normalized >= margin).all() and (normalized <= 1 - margin).all()
            image_path = capture_dir / input_view['image']
            assert digest(image_path) == input_view['image_sha256']
            with Image.open(image_path) as source:
                rgb = source.convert('RGB')
            assert rgb.size == tuple(input_view['image_size_wh']) == (640, 1100)
            assert np.asarray(rgb).std() > 1.
            mask = np.asarray(Image.open(capture_dir / 'evaluation' / (side + '_torch_mask.png')))
            assert np.count_nonzero(mask) == capture['torch_pixels'][side] == 0
            stereo_draw.text((8 + eye_index * 648, 8), f"{view['station']} / {side}", fill='#14212c', font=font)
            stereo.paste(rgb, (8 + eye_index * 648, 40))
            preview.paste(rgb.resize((240, 413), Image.Resampling.LANCZOS), (8 + i * 496 + eye_index * 240, 380))
            eyes.append({'eye': side, 'vertices_checked': len(vertices),
                         'all_vertices_in_frame_with_margin': True,
                         'normalized_bounds': [normalized.min(0).tolist(), normalized.max(0).tolist()],
                         'minimum_edge_margin_fraction': float(min(normalized.min(), 1 - normalized.max())),
                         'torch_pixels': 0, 'image_sha256': input_view['image_sha256']})
        assert not view['penetrations_m']
        assert view['foot_floor_max_error_m'] < .001 and view['com_support_margin_m'] > 0
        stereo.save(images / (view['station'] + '_stereo.png'))
        with Image.open(capture_dir / 'robot_context.png') as source:
            context = source.convert('RGB')
        assert context.size == (960, 640) and np.asarray(context).std() > 1
        preview.paste(context.resize((480, 320), Image.Resampling.LANCZOS), (8 + i * 496, 36))
        draw.text((8 + i * 496, 4), f"{view['station']}  robot gap {gap:.2f} m", fill='#14212c', font=font)
        draw.text((8 + i * 496, 358), f'Left                     Right   |   camera {distance:.2f} m', fill='#14212c', font=small)
        rows.append({'station': view['station'], 'camera_to_center_m': distance,
                     'robot_base_center_to_workpiece_xy_m': gap,
                     'rendered_baseline_m': float(baseline), 'actual_pose_max_element_error': pose_error,
                     'eyes': eyes, 'static_supported_pose': True, 'usd': str(capture_dir / 'arm_pose.usda')})
    control = json.loads((folder / 'captures' / plan['views'][0]['id'] / 'evaluation/torch_segmentation_control.json').read_text())
    assert control['torch_pixels'] > 0
    preview.save(images / 'overview.png')
    validation = {'completed': True, 'stereo_pairs': len(rows), 'eyes_validated': len(rows) * 2,
                  'image_resolution_wh': [640, 1100], 'required_edge_margin_fraction': margin,
                  'source_usdc_unchanged': True, 'intrinsics_unchanged': True,
                  'torch_segmentation_positive_control_pixels': control['torch_pixels'],
                  'physics_executed': False, 'stations': rows}
    (folder / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    lines = ['# 双目完整观察工件的机器狗站位', '',
             'A15、A03、A21 已分别保存机器狗底座与机械臂的新静态位姿，并在 Isaac Sim 实际采集左右目。',
             '保持原相机内参、640×1100 分辨率和 90 mm 基线。两块工件板件的全部顶点在每只眼中均留有至少 8% 画幅余量，六张照片的焊枪像素均为 0。', '',
             '| 工位 | 机器狗底座中心至工件前沿的水平距离 | 相机光学中心至工件中心的直线距离 |',
             '|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['station']} | {row['robot_base_center_to_workpiece_xy_m']:.2f} m | {row['camera_to_center_m']:.2f} m |")
    lines += ['', '距离以底板前沿及两板组成的工件中心为参照，不是狗身外壳至桌沿的净空。', '',
              '![机器狗站位与双目画面](images/overview.png)', '',
              '以下为完整左右目原图并排展示。完整入画表示工件不被画框裁切；夹具与板件自身的遮挡仍然存在。', '']
    for view in plan['views']:
        lines += [f"## {view['station']}", '',
                  f"[在 Isaac Sim 打开此站位 USD](captures/{view['id']}/arm_pose.usda)", '',
                  f"![{view['station']} 左右目原图](images/{view['station']}_stereo.png)", '']
    lines += ['当前约 1.9–2.0 m 的相机距离用于完整观察。HJ-90 文档标称测量工作距离为 300–800 mm，后续精细焊缝测量仍需靠近后分区域拍摄。', '',
              '本次只调整并验证取景；未运行新的 DA3、SAM3 或焊缝误差评价。USD 是独立静态站位，未执行行走或机械臂运动轨迹。', '',
              '复现命令（第一步使用新的空输出目录）：', '', '```bash',
              '.venv/bin/python scripts/prepare_full_station_views.py --output-dir outputs/benchmarks/full_frame_new',
              '.venv/bin/python scripts/capture_station_overviews.py --resume --output-dir outputs/benchmarks/full_frame_new --gpu 0',
              '.venv/bin/python scripts/report_full_station_views.py --output-dir outputs/benchmarks/full_frame_new',
              '```', '']
    (folder / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'completed': True, 'report': str(folder / 'README.md'),
                      'pairs': len(rows), 'torch_pixels_all_eyes': 0}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    report(parser.parse_args().output_dir.resolve())
