import time

from loguru import logger
from rich.console import Console

last_called = None

def has_been_called_this_second():
    global last_called
    current_time = time.time()
    current_second = int(current_time)
    if last_called == current_second:
        return True
    else:
        last_called = current_second
        return False

def log_formatter(record: dict) -> str:
    """Log message formatter"""

    color_map = {
        'TRACE': 'dim blue',
        'DEBUG': 'cyan',
        'INFO': 'blue',
        'SUCCESS': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'white on red'
    }

    color = color_map.get(record['level'].name)

    level_map = {
        'TRACE'   : f'   [{color}]{{level}}[/{color}]',
        'DEBUG'   : f'   [{color}]{{level}}[/{color}]',
        'INFO'    : f'    [{color}]{{level}}[/{color}]',
        'SUCCESS' : f' [{color}]{{level}}[/{color}]',
        'WARNING' : f' [{color}]{{level}}[/{color}]',
        'ERROR'   : f'   [{color}]{{level}}[/{color}]',
        'CRITICAL': f'[{color}]{{level}}[/{color}]'
    }
    level = level_map.get(record['level'].name)
    printtime = "                   " if has_been_called_this_second() else "[u][dim cyan]{time:YYYY/MM/DD HH:mm:ss}[/dim cyan][/u]"
    return (
        printtime+'  '+level+' | - {message}'
    )

console = Console()
logger.remove()
logger.add(
    console.print,
    format=log_formatter,
    colorize=True,
)