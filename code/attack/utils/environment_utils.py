import multiprocessing
import os
import platform
import socket

import torch


def is_windows_os() -> bool:
    return platform.system() == 'Windows'


def is_linux_os() -> bool:
    return platform.system() == 'Linux'


def get_machine_name() -> str:
    if is_linux_os():
        return socket.gethostname()
    else:
        return os.environ['COMPUTERNAME']


def get_num_workers(dataset: torch.utils.data.Dataset, batch_size: int) -> int:
    if is_windows_os():
        return 0
    num_cores: int = multiprocessing.cpu_count()
    num_batches: int = max(1, (len(dataset) + batch_size - 1) // batch_size)
    return min(num_cores - 2, num_batches)
