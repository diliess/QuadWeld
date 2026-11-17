"""Verify the stereo entry point, capture workers and saved-data replay."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main


class PipelineEntryTests(unittest.TestCase):
    def test_stereo_runs_scene_and_sam_workers_for_the_selected_station(self):
        def capture_worker(command, _):
            output = Path(command[command.index('--output-dir') + 1])
            output.mkdir()
            plan = json.loads(Path(command[command.index('--pose-plan') + 1]).read_text())
            (output / 'views.json').write_text(json.dumps({'views': plan['views']}))
            self.assertIn('--rgb-only', command)

        def worker(command, log):
            if command[2] == 'weld_scene.stereo_capture':
                capture_worker(command, log)
            else:
                self.assertEqual(command[2], 'model_runtime.sam_adapter')
                self.assertIn('A26_', command[3])

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'run'
            collision = Path(folder) / 'collision.xml'
            collision.touch()
            overview = {'views': [{'station': 'A26', 'id': 'A26_overview'}]}
            close = {'views': [{'station': 'A26', 'id': 'A26_sweep_00'}]}
            observed = {'estimated_range_m': 2.0}
            ready = {'ready': True, 'issues': [], 'interpreters': {'isaac': 'isaac', 'sam': 'sam'}}
            with (
                patch('main.runtime.preflight', return_value=ready) as preflight,
                patch('main.runtime.run_process', side_effect=worker) as launch,
                patch('weld_scene.sweep_planning.plan_overview', return_value=overview) as plan_overview,
                patch('weld_vision.coarse_localization.measure_overview', return_value=observed),
                patch('weld_scene.sweep_planning.plan_close', return_value=close) as plan_close,
                patch('weld_vision.multiview_localization.measure', return_value={'fused': {'A26': {'status': 'ok'}}}) as measure,
            ):
                code = main.main(['--station', 'A26', '--collision-model', str(collision), '--output-dir', str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(preflight.call_args.args[1], ('isaac', 'sam'))
            self.assertEqual([c.args[0][2] for c in launch.call_args_list],
                             ['weld_scene.stereo_capture', 'model_runtime.sam_adapter'] * 2)
            plan_overview.assert_called_once_with(['A26'], collision)
            plan_close.assert_called_once_with(overview['views'], {'A26': observed}, collision)
            self.assertEqual(measure.call_args.kwargs['station'], 'A26')
            self.assertEqual(json.loads((output / 'run_report.json').read_text())['status'], 'ok')

    def make_replay(self, folder, station='A26'):
        source = folder / 'saved'
        overview = {'station': station, 'id': station + '_overview'}
        close = {'station': station, 'id': station + '_sweep_00', 'view': 'sweep'}
        for phase, view in (('overview', overview), ('close', close)):
            capture = source / phase / 'captures' / view['id']
            capture.mkdir(parents=True)
            (capture / 'input.json').write_text('{}')
            masks = source / phase / 'sam'
            masks.mkdir()
            (masks / (view['id'] + '.npz')).touch()
            (source / phase / 'views.json').write_text(json.dumps({'views': [view]}))
        (source / 'close_plan.json').write_text(json.dumps({'views': [close], 'overview': {station: {}}}))
        return source

    def test_default_station_and_removed_options(self):
        args = main.parse_args([])
        self.assertEqual(args.station, 'A25')
        self.assertEqual(args.min_views, 2)
        self.assertEqual(args.output_dir.parent, main.ROOT / 'outputs/runs')
        self.assertRegex(args.output_dir.name, r'^\d{8}_\d{6}_\d{6}__stereo__A25_front__sweep$')
        replay = main.parse_args(['--station', 'A26', '--replay', '/tmp/saved_stereo'])
        self.assertRegex(replay.output_dir.name, r'^\d{8}_\d{6}_\d{6}__stereo__A26_front__replay$')
        retired = [
            ['--pipeline', 'da3'], ['--pipeline', 'stereo'], ['--image', 'frame.png'],
            ['--intrinsics', 'camera.json'], ['--da3-python', '/python'],
            ['--model-cache-dir', '/cache'], ['--sensor', 'hj90'], ['--capture-only'],
            ['--rgb-profile', 'vendor'], ['--rgb-resolution', '640', '480'],
            ['--scan-lines', '640'], ['--_capture-worker'],
        ]
        for argv in retired:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main.parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_invalid_stereo_options_are_rejected(self):
        for argv in (['--station', 'A27'], ['--min-views', '0'], ['--gpu', '-1'],
                     ['--semantic-threshold', '0'], ['--semantic-threshold', '1.1']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main.parse_args(argv)

    def test_replay_uses_saved_inputs_without_starting_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_replay(Path(folder))
            output = Path(folder) / 'new'
            observed = {'estimated_range_m': 2.0}
            with (
                patch('main.runtime.preflight') as preflight,
                patch('main.runtime.run_process') as launch,
                patch('weld_vision.coarse_localization.measure_overview', return_value=observed),
                patch('weld_vision.multiview_localization.measure', return_value={
                    'fused': {'A26': {'status': 'ok', 'root_world_m': []}}}) as measure,
            ):
                code = main.main(['--station', 'A26', '--replay', str(source),
                                  '--min-views', '3', '--output-dir', str(output)])
            self.assertEqual(code, 0)
            preflight.assert_not_called()
            launch.assert_not_called()
            self.assertEqual(measure.call_args.args, (source, output / 'results', 3))
            self.assertEqual(measure.call_args.kwargs['plan']['overview']['A26'], observed)
            report = json.loads((output / 'run_report.json').read_text())
            self.assertTrue(report['completed'])
            self.assertEqual(report['pipeline'], 'stereo')
            self.assertEqual(report['source'], str(source))
            self.assertEqual(report['fused'], 'results/A26/fused.json')

    def test_replay_check_does_not_create_output_or_start_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_replay(Path(folder))
            output = Path(folder) / 'new'
            with patch('main.runtime.preflight') as preflight, patch('main.execute_stereo') as execute:
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    code = main.main(['--station', 'A26', '--replay', str(source),
                                      '--output-dir', str(output), '--check'])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(stdout.getvalue())['ready'])
            preflight.assert_not_called()
            execute.assert_not_called()
            self.assertFalse(output.exists())

    def test_replay_check_rejects_missing_segmentation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_replay(Path(folder))
            (source / 'close/sam/A26_sweep_00.npz').unlink()
            with self.assertRaises(FileNotFoundError):
                main.main(['--station', 'A26', '--replay', str(source), '--check'])

    def test_existing_results_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_replay(Path(folder))
            output = Path(folder) / 'new'
            output.mkdir()
            existing = output / 'result.json'
            existing.write_text('existing result')
            with patch('main.execute_stereo') as execute:
                with self.assertRaisesRegex(ValueError, 'empty output directory'):
                    main.main(['--station', 'A26', '--replay', str(source), '--output-dir', str(output)])
            execute.assert_not_called()
            self.assertEqual(existing.read_text(), 'existing result')

    def test_measurement_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_replay(Path(folder))
            output = Path(folder) / 'new'
            with patch('weld_vision.coarse_localization.measure_overview', side_effect=ValueError('bad stereo pair')):
                with self.assertRaisesRegex(ValueError, 'bad stereo pair'):
                    main.main(['--station', 'A26', '--replay', str(source), '--output-dir', str(output)])
            report = json.loads((output / 'run_report.json').read_text())
            self.assertFalse(report['completed'])
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error'], 'bad stereo pair')

    def test_insufficient_views_returns_nonzero_and_preserves_status(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_replay(Path(folder))
            output = Path(folder) / 'new'
            with (
                patch('weld_vision.coarse_localization.measure_overview', return_value={}),
                patch('weld_vision.multiview_localization.measure', return_value={
                    'fused': {'A26': {'status': 'insufficient_views'}}}),
            ):
                code = main.main(['--station', 'A26', '--replay', str(source), '--output-dir', str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads((output / 'run_report.json').read_text())['status'], 'insufficient_views')


if __name__ == '__main__':
    unittest.main()
