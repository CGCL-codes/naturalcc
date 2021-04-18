# -*- coding: utf-8 -*-

import math
import torch
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


ACT2FN = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "softmax": F.softmax,
    "log_softmax": F.log_softmax,
    "leaky_relu": F.leaky_relu,

    "gelu": F.gelu,
    "swish": swish,
    "gelu_new": gelu_new,
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            f"Function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())} or torch.nn.functional. " \
            f"Or you can register new activations at {__file__}"
        )
