# -*- coding: utf-8 -*-

from typing import *

from pathos.multiprocessing import (
    ProcessingPool as Pool,
    cpu_count
)

from ncc import LOGGER


class PPool:
    """pathos multi-processing pool"""

    def __init__(self, processor_num: int = None, ):
        self.processor_num = cpu_count() if processor_num is None \
            else min(processor_num, cpu_count())
        LOGGER.debug('Building Pathos multi-processing pool with {} cores.'.format(self.processor_num))
        self._pool = Pool(self.processor_num)

    def flatten_params(self, params: List):
        """params: List[*args, **kwargs]"""
        # block_size = int(math.ceil(len(params) / self.processor_num))
        # block_num = int(math.ceil(len(params) / block_size))
        block_size = (len(params) + self.processor_num - 1) // self.processor_num
        block_num = (len(params) + block_size - 1) // block_size
        block_params = [params[i * block_size:(i + 1) * block_size] for i in range(block_num)]
        return block_params

    def close(self):
        self._pool.close()
        self._pool.join()
        self._pool.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def feed(self, func: Any, params: List, one_params: bool = False) -> List[Any]:
        if one_params:
            result = self._pool.amap(func, params).get()
        else:
            params = tuple(zip(*params))
            result = self._pool.amap(func, *params).get()
        return result


__all__ = [
    'PPool',
]

if __name__ == '__main__':
    add = lambda x, y: x + y

    params = [
        [1, 2],
        [1, 2],
    ]

    same = lambda x: x
    params = [
        [1, ],
        [2, ],
    ]

    with PPool(3) as pool:
        out = pool.feed(same, params)
        print(out)
