import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from config import SCENE
from robot.robot_controller import RobotController


class ControllerFailureTests(unittest.TestCase):
    def test_execution_failure_holds_and_does_not_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = object.__new__(RobotController)
            controller.output_dir = Path(directory)
            controller.config = SCENE
            controller.history = []
            controller.progress_callback = None
            controller.robot = Mock()
            controller.indices = list(range(6))
            measured = np.full(6, 0.1)
            controller.current_q = Mock(return_value=measured)
            controller.move_to_pose = Mock(side_effect=RuntimeError("Tracking diverged"))
            joints = np.zeros((3, 6))
            poses = np.repeat(np.eye(4)[None], 3, axis=0)
            with self.assertRaisesRegex(RuntimeError, "Tracking diverged"):
                controller.follow_cartesian_path(joints, poses, ["home", "trace", "retreat"])
            self.assertEqual(controller.move_to_pose.call_count, 1)
            np.testing.assert_equal(controller.robot.set_dof_position_targets.call_args.args[0], measured)
            report = json.loads((controller.output_dir / "execution_report.json").read_text())
            self.assertFalse(report["completed"])
            self.assertEqual(report["status"], "failed")
            self.assertTrue((controller.output_dir / "tcp_execution.csv").is_file())


if __name__ == "__main__":
    unittest.main()
