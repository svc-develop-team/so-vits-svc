import time
import logger

def work(i):
    # logger.info("[red]Working[/red]: {}", i)
    time.sleep(0.05)

def main():
    logger.info("Before")
    with logger.Progress() as progress:
        for i in progress.track(range(100)):
            work(i)
    logger.info("After")

if __name__ == "__main__":
    main()