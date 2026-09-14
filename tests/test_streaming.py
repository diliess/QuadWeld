import unittest
from unittest.mock import Mock, patch

from scene.streaming import StreamingPanel


class StreamingGateTests(unittest.TestCase):
    def panel(self):
        panel = object.__new__(StreamingPanel)
        panel.started = False
        panel.button = Mock()
        panel.label = Mock()
        return panel

    @patch('scene.streaming.sys.stdin')
    def test_wait_pumps_frames_until_user_starts(self, stdin):
        stdin.isatty.return_value = False
        panel, app = self.panel(), Mock()
        app.is_running.return_value = True
        def update():
            if app.update.call_count == 3:
                panel.start()
        app.update.side_effect = update
        panel.wait_for_start(app)
        self.assertEqual(app.update.call_count, 3)
        self.assertTrue(panel.started)
        self.assertFalse(panel.button.enabled)

    def test_closing_does_not_start_task(self):
        panel, app = self.panel(), Mock()
        app.is_running.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'closed before task start'):
            panel.wait_for_start(app)
        self.assertFalse(panel.started)


if __name__ == '__main__':
    unittest.main()
