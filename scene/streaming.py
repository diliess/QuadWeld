"""Start gate and task status within the streamed Isaac Sim application."""

import select
import sys


class StreamingPanel:
    def __init__(self, *, preview=False):
        import omni.ui as ui

        self.started = False
        self.window = ui.Window("QuadWeld Task", width=340, height=160)
        with self.window.frame:
            with ui.VStack(spacing=8):
                self.label = ui.Label("Ready. Connect your client, then start.", word_wrap=True)
                self.button = ui.Button("Capture RGB-D" if preview else "Start task", clicked_fn=self.start, height=40)
                ui.Label("Workpiece preview and RGB-D capture" if preview
                         else "Capture > Plan > Approach > Trace > Retreat", word_wrap=True)

    def start(self):
        self.started = True
        self.button.enabled = False
        self.set_status("Starting task...")

    def set_status(self, message):
        self.label.text = message

    def wait_for_start(self, app):
        print("WebRTC ready. Connect the Windows client, then click the start/capture button "
              "in 'QuadWeld Task', or press Enter in this terminal.", flush=True)
        while not self.started:
            if not app.is_running():
                raise RuntimeError("Application closed before task start")
            # Keep Kit/UI/rendering alive while waiting; input() would freeze the stream.
            app.update()
            if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.readline():
                    self.start()
