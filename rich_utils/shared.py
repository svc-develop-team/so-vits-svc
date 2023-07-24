import atexit

from rich.live import Live

import rich_utils

live = Live(refresh_per_second=10, transient=True)

live.start()

def on_exit():
    live.stop()

atexit.register(on_exit)

progress = rich_utils.MProgress("PreProcessing F0 and Hubert","Workers")

live.update(progress)