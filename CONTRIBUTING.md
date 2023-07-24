# 写给贡献者
## 仓库日志

仓库提供了 `Logger`(暂未实装) 与 `Progress` 两个类, 用于项目的日志与进度条.

`Progress` 的简易示例如下


多条
```python
from time import sleep
import rich_utils
from rich.live import Live

progress = rich_utils.MProgress("PreProcessing F0 and Hubert","Workers")
task1 = progress.add_task(100, "Worker 1")
task2 = progress.add_task(400, "Worker 2")

with Live(progress, refresh_per_second=10, transient=True) as live:
    while not progress.overall_progress.finished:
        progress.update(task1)
        progress.update(task2)
        sleep(0.004)
```