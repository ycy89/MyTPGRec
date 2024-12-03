import logging
import datetime
import os

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def init_logger(name):
    logger = logging.getLogger()
    logger.setLevel(level = logging.INFO)
    os.makedirs('./log', exist_ok=True)
    handler = logging.FileHandler(f'./log/{name}-{get_local_time()}.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    # logging.basicConfig(level=logging.INFO, handlers=[handler, console])

    return logger



