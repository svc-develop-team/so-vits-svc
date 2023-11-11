from loguru import logger

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress as _Progress
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import time

import os
import datetime

console = Console(stderr=None)

logger.remove()


def format_level(str, length):
    if len(str) < length:
        str = str + " " * (length - len(str))
    else:
        str = str
    # 给 str 上对应 level 的颜色
    if str == "INFO   ":
        str = f"[bold green]{str}[/bold green]"
    elif str == "WARNING":
        str = f"[bold yellow]{str}[/bold yellow]"
    elif str == "ERROR  ":
        str = f"[bold red]{str}[/bold red]"
    elif str == "DEBUG  ":
        str = f"[bold cyan]{str}[/bold cyan]"
    return str

def default_format(record):
    # print(record)
    return f"[green]{record['time'].strftime('%Y-%m-%d %H:%M:%S')}[/green] | [level]{format_level(record['level'].name,7)}[/level] | [cyan]{record['file'].path.replace(os.getcwd()+os.sep,'')}:{record['line']}[/cyan] - [level]{record['message']}[/level]\n"


logger.add(lambda m: console.print(m, end=""), format=default_format, colorize=True)


def addLogger(path):
    logger.add(
        path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}",
        colorize=True,
    )


info = logger.info
error = logger.error
warning = logger.warning
warn = logger.warning
debug = logger.debug


def Progress():
    return _Progress(
        TextColumn("[progress.description]{task.description}"),
        # TextColumn("[progress.description]W"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("[red]*Elapsed[/red]"),
        TimeElapsedColumn(),
        console=console,
    )
