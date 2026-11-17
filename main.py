"""Select a welding station and run the calibrated stereo weld scan."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from model_runtime import runtime
from weld_scene.scene import ROOT, write_json

STEREO_STATIONS = ('A25', 'A26')

def stereo_inputs(directory, views):
    return [directory / 'captures' / view['id'] / 'input.json' for view in views]


def capture_and_segment(args, checked, phase, plan):
    output = args.output_dir
    directory = output / phase
    plan_path = output / f'{phase}_plan.json'
    write_json(plan_path, plan)
    interpreters = checked['interpreters']
    runtime.run_process([interpreters['isaac'], '-m', 'weld_scene.stereo_capture',
                         '--pose-plan', str(plan_path), '--output-dir', str(directory),
                         '--gpu', str(args.gpu), '--rgb-only'], output / f'{phase}_capture.log')
    views = json.loads((directory / 'views.json').read_text())['views']
    if [v['id'] for v in views] != [v['id'] for v in plan['views']]:
        raise ValueError(f'{phase} capture does not match the scan plan')
    pairs = stereo_inputs(directory, views)
    runtime.run_process([interpreters['sam'], '-m', 'model_runtime.sam_adapter', *map(str, pairs),
                         '--output-dir', str(directory / 'sam'), '--gpu', str(args.gpu),
                         '--threshold', str(args.semantic_threshold)], output / f'{phase}_sam.log')
    return views


def replay_inputs(run, station):
    manifest = run / 'overview' / 'views.json'
    views = [v for v in json.loads(manifest.read_text())['views'] if v['station'] == station]
    plan = json.loads((run / 'close_plan.json').read_text())
    close = json.loads((run / 'close' / 'views.json').read_text())['views']
    if [v['id'] for v in close] != [v['id'] for v in plan['views']]:
        raise ValueError('Close capture does not match the scan plan')
    selected = [v for v in close if v['station'] == station]
    if len(views) != 1 or not any(v['view'] == 'sweep' for v in selected):
        raise ValueError(f'Replay data for {station} must include one overview and a close sweep')
    for phase, records in (('overview', views), ('close', selected)):
        for pair in stereo_inputs(run / phase, records):
            mask = run / phase / 'sam' / f'{pair.parent.name}.npz'
            if not pair.is_file() or not mask.is_file():
                raise FileNotFoundError(f'Missing stereo input or SAM candidates for {pair.parent.name}')
    return views, plan


def execute_stereo(args, checked):
    import cv2

    from weld_scene import sweep_planning
    from weld_vision import coarse_localization, multiview_localization

    cv2.setNumThreads(1)
    output = args.output_dir
    report_path = output / 'run_report.json'
    source = args.replay or output
    report = {'pipeline': 'stereo', 'station': args.station, 'source': str(source), 'completed': False}
    try:
        report['status'] = 'overview'
        write_json(report_path, report)
        if args.replay:
            overview, close_plan = replay_inputs(source, args.station)
        else:
            overview_plan = sweep_planning.plan_overview([args.station], args.collision_model)
            overview = capture_and_segment(args, checked, 'overview', overview_plan)
        observation = coarse_localization.measure_overview(source, overview[0])
        report.update(status='close', overview=observation)
        write_json(report_path, report)
        if args.replay:
            close_plan['overview'][args.station] = observation
        else:
            close_plan = sweep_planning.plan_close(overview, {args.station: observation}, args.collision_model)
            capture_and_segment(args, checked, 'close', close_plan)
        result = multiview_localization.measure(source, output / 'results', args.min_views,
                                               plan=close_plan, station=args.station)
        fused = result['fused'][args.station]
        report.update(completed=True, status=fused['status'], predictions='results/predictions.json')
        if 'root_world_m' in fused:
            report['fused'] = f'results/{args.station}/fused.json'
        write_json(report_path, report)
        print(f"Completed: {report['status']}\nResults: {output}", flush=True)
        return 0 if fused['status'] == 'ok' else 2
    except BaseException as error:
        report.update(completed=False, status='failed', error=str(error))
        write_json(report_path, report)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--station', choices=STEREO_STATIONS, default=STEREO_STATIONS[0], help='Stereo station, default A25')
    parser.add_argument('--replay', type=Path, help='Replay stereo captures and saved SAM candidates')
    parser.add_argument('--collision-model', type=Path, help='MuJoCo collision XML for live stereo planning')
    parser.add_argument('--min-views', type=int, default=2, help='Minimum supporting views for a fused toe point')
    parser.add_argument("--output-dir", type=Path, help="New, empty run directory")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index")
    parser.add_argument("--headless", action="store_true", help="Run without a GUI (stereo capture is always headless)")
    parser.add_argument("--isaac-python", type=Path, help="Isaac Sim Python executable")
    parser.add_argument("--sam-python", type=Path, help="SAM3 Python executable")
    parser.add_argument("--semantic-threshold", type=float, default=.25)
    parser.add_argument("--check", action="store_true", help="Check dependencies or replay inputs")
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.min_views < 1 or not 0 < args.semantic_threshold <= 1:
        parser.error("Use --gpu >= 0, --min-views >= 1 and 0 < --semantic-threshold <= 1")
    stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S_%f")
    purpose = 'replay' if args.replay else 'sweep'
    args.output_dir = args.output_dir or ROOT / 'outputs/runs' / f'{stamp}__stereo__{args.station}_front__{purpose}'
    for field in ('output_dir', 'replay', 'collision_model'):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).expanduser().resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.replay:
        replay_inputs(args.replay, args.station)
        checked = {'ready': True, 'interpreters': {}, 'issues': []}
    else:
        from weld_scene.sweep_planning import COLLISION_MODEL

        args.collision_model = args.collision_model or COLLISION_MODEL.resolve()
        checked = runtime.preflight(args, ('isaac', 'sam'))
        if not args.collision_model.is_file():
            checked['issues'].append(f'Missing collision model: {args.collision_model}')
            checked['ready'] = False
    checked.update(station=args.station, pipeline='stereo')
    if args.check:
        print(json.dumps(checked, ensure_ascii=False, indent=2))
        return 0 if checked['ready'] else 1
    if not checked['ready']:
        raise RuntimeError("\n".join(checked['issues']))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Use an empty output directory to avoid mixing runs: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return execute_stereo(args, checked)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"QuadWeld failed: {error}", file=sys.stderr)
        raise SystemExit(1)
