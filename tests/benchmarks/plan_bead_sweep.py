"""Plan a mounted sweep from the command line, including comparison views."""

import argparse
import json
from pathlib import Path

import cv2

from weld_scene.scene import write_json
from weld_scene.sweep_planning import COLLISION_MODEL, STATIONS, plan_close, plan_overview
from weld_vision.sweep import measure_overview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan-overview', 'plan-close'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stations', nargs='+', default=STATIONS, choices=STATIONS)
    parser.add_argument('--collision-model', type=Path, default=COLLISION_MODEL)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    collision = args.collision_model.resolve()
    cv2.setNumThreads(1)
    if args.stage == 'plan-close':
        views = json.loads((args.run_dir / 'overview' / 'views.json').read_text())['views']
        observations = {v['station']: measure_overview(args.run_dir, v) for v in views}
        plan = plan_close(views, observations, collision, range_checks=True)
        name = 'close_plan.json'
    else:
        plan = plan_overview(args.stations, collision)
        name = 'overview_plan.json'
    path = args.run_dir / name
    write_json(path, plan)
    print('BEAD_SCAN_PLAN', len(plan['views']), path)


if __name__ == "__main__":
    main()
