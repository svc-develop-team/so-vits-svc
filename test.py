import time
import logger

def work(i):
    # logger.info("[red]Working[/red]: {}", i)
    # time.sleep(0.)
    pass

def main():
    logger.info("Before")
    with logger.Progress() as progress:
        for i in progress.track(range(100), description="Workkkkkking"):
            work(i)
    logger.info("After")
    logger.warning("Warning")
    logger.error("Error")
    logger.debug("Debug")
    logger.info("Info")

if __name__ == "__main__":
    main()