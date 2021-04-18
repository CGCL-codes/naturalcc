# -*- coding: utf-8 -*-

import random
import torch
import numpy as np
import contextlib


def set_seed(seed):
    # Set seed based on args.seed and the update number so that we get
    # reproducible results when resuming from checkpoints
    assert isinstance(seed, int)
    random.seed(seed)
    np.random.seed(seed)
    set_torch_seed(seed)


def set_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@contextlib.contextmanager
def with_torch_seed(seed):
    assert isinstance(seed, int)
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    set_torch_seed(seed)
    yield
    torch.set_rng_state(rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
