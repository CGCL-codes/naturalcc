# -*- coding: utf-8 -*-

import functools
import torch
import os
from ncc import LOGGER


def pre(*pargs, **pkwargs):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            LOGGER.info(f"PID: {os.getppid()} - {os.getpid()}")
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@pre()
def foo(device_id):
    LOGGER.info(f"GPU device id: {device_id}")


if __name__ == '__main__':
    for i in range(4):
        foo(i)
