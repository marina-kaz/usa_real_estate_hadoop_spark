import psutil
from time import time
from tap import Tap
import logging
from datetime import datetime

class CLI(Tap):
    pid: int
    log_prefix: str = "2nodes"


def get_logger(prefix: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.flush()

    file_handler = logging.FileHandler(f'/opt/spark/apps/{prefix}_{datetime.now()}.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    file_handler.flush()

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def monitor_memory(pid, log_prefix, interval=5):
    logger = get_logger(log_prefix + "_memory")
    # process = psutil.Process(pid=pid)
    while True:
        try:
            process = psutil.Process(pid)
            with process.oneshot():
                ram_bytes_used = process.memory_info().rss  # Resident set size = physical RAM
                children = process.children(recursive=True)
                for child in children:
                    with child.oneshot():
                        ram_bytes_used += child.memory_info().rss
            logger.info(f'RSS: {ram_bytes_used.rss}')
            time.sleep(interval)
        except Exception as ex:
            logger.info(ex)
            print(ex)



if __name__ == "__main__":
    args = CLI(underscores_to_dashes=True).parse_args()
    monitor_memory(args.pid, args.log_prefix)
