from loguru import logger

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress as _Progress

console = Console(stderr=None)
# logger.remove()
# handler = RichHandler(console=console)
# # logger.add(
# #     # lambda _: console.print(_, end=''),
# #     handler,
# #     level='TRACE',
# #     # format=log_formater,
# #     format=    "[green]{time:YYYY-MM-DD HH:mm:ss.SSS}[/green] | "
# #     "[level]{level: <8}[/level] | "
# #     "[cyan]{name}[/cyan]:[cyan]{function}[/cyan]:[cyan]{line}[/cyan] - [level]{message}[/level]",
# #     colorize=True,
# # )
# logger.add(handler, colorize=True)

logger.remove()  # Remove default 'stderr' handler

# We need to specify end=''" as log message already ends with \n (thus the lambda function)
# Also forcing 'colorize=True' otherwise Loguru won't recognize that the sink support colors
logger.add(lambda m: console.print(m, end=""),
            format="[green]{time:YYYY-MM-DD HH:mm:ss.SSS}[/green] | "
                   "[level]{level: <8}[/level] | "
                   "[cyan]{name}[/cyan]:[cyan]{function}[/cyan]:[cyan]{line}[/cyan] - [level]{message}[/level]",
            colorize=True)
info = logger.info
error = logger.error
warning = logger.warning
debug = logger.debug

def Progress():
    return _Progress(console=console)