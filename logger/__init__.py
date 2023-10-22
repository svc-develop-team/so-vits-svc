from loguru import logger

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress as _Progress
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

import time

console = Console(stderr=None)

logger.remove()

logger.add(lambda m: console.print(m, end=""),
            format="[green]{time:YYYY-MM-DD HH:mm:ss.SSS}[/green] | "
                   "[level]{level: <8}[/level] | "
                   "[cyan]{name}[/cyan]:[cyan]{function}[/cyan]:[cyan]{line}[/cyan] - [level]{message}[/level]",
            colorize=True)

def addLogger(path):
       logger.add(path, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                                   "{level: <8} | "
                                   "{name}:{function}:{line} - {message}",
                                   colorize=True)

info = logger.info
error = logger.error
warning = logger.warning
debug = logger.debug

def Progress():
    return _Progress(
              TextColumn("[progress.description]{task.description}"),
              # TextColumn("[progress.description]W"),
              BarColumn(),
              TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
              TimeRemainingColumn(),
              TextColumn("[red]*Elapsed[/red]"),
              TimeElapsedColumn(),console=console)