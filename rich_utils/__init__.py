import random
import sys
from typing import Iterable

from rich import color
from rich.console import Console, ConsoleOptions, ConsoleRenderable, Group, RenderResult
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn
from rich.table import Table
from rich.text import Text

if sys.version_info >= (3, 11):
    from typing import Protocol
else:
    from typing_extensions import Protocol

class CanGetKeys(Protocol):
    def keys(self) -> Iterable:
        pass

def generate_array_from_dict_keys(a: CanGetKeys):
    if not (hasattr(a, 'keys') and callable(a.keys)):
        raise ValueError("Input must be able to .keys()")
    
    return tuple(a.keys())

# def random_color() -> str:
#     return random.choice(generate_array_from_dict_keys(color.ANSI_COLOR_NAMES))

def random_color_text(text: str = "") -> Text:
    return f"[{random.choice(generate_array_from_dict_keys(color.ANSI_COLOR_NAMES))}]{text}"

class MProgress(ConsoleRenderable):
    def __init__(self,MainTitle,SubTitle):

        self.MainTitle = MainTitle
        self.SubTitle = SubTitle

        self.job_progress = Progress()

        self.overall_progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
        )

        self.progress_text = Text("Processing: 0/0", justify="center")

        self.overall_task = self.overall_progress.add_task("All") # 这边 init 的时候什么task都没有的说

        self.progress_table = Table.grid()

        self.progress_table.add_row(
            Panel(
                Group(
                    self.overall_progress,
                    self.progress_text
                ), title=self.MainTitle, border_style="green", padding=(5, 5), height=14
            ),
            Panel(self.job_progress, title=f"[b]{self.SubTitle}", border_style="red", padding=(2, 2), height=14)
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult: # 刷新的时候计算 overall
        total = sum(task.total for task in self.job_progress.tasks)
        completed = sum(task.completed for task in self.job_progress.tasks)
        self.overall_progress.update(
            self.overall_task, 
            total=total,
            completed=completed
        )
        yield self.progress_table

    def add_task(self, total: int, description: str = "", idx: int = None):
        if not idx:
            if len(self.job_progress.task_ids) != 0:
                idx = self.job_progress.task_ids[-1]
            else: 
                idx = 0

        return self.job_progress.add_task(random_color_text(description), total=total)

    def update(self, TaskID: TaskID, value: int = -1):
        total = sum(task.total for task in self.job_progress.tasks)
        completed = sum(task.completed for task in self.job_progress.tasks)
        # 更新 progress_text
        self.progress_text._text = [f"Processing: {completed}/{total}"]
        # self.progress_text.apply_meta({"_text": f"Processing: {completed}/{total}"})
        # self.progress_table.rows[0].title = "6"
        # print(777)
        if value == -1:
            # log.logger.debug("No value")
            # value = self.job_progress.tasks[TaskID].completed + 1
        # if not self.job_progress.tasks[TaskID].finished:
            self.job_progress.advance(TaskID)
        # else:
        else:
            self.job_progress.update(TaskID, completed=value)
        if self.job_progress.tasks[TaskID].finished:
            self.job_progress.update(TaskID, visible=False)
            # self.job_progress.remove_task(TaskID)
            # 如果所有任务都完成了，就把 progress_table 隐藏掉
            # if all([task.finished for task in self.job_progress.tasks]):
            #     self.progress_table.rows = []