import logging
import random
import torch

specials = {'bert': '#', 'gpt2': 'Ġ', 'xlnet': '▁', 'roberta': 'Ġ'}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def select_indices(tokens, raw_tokens, model, mode):
    mask = []
    raw_i = 0
    collapsed = ''
    if model=='microsoft/codebert-base' or model=='microsoft/graphcodebert-base':
      model = 'roberta'
    special = specials[model]

    for i in range(len(tokens)):
        token = tokens[i]

        while len(token) > 0 and token[0] == special:
            token = token[1:]
        if collapsed == '' and len(token) > 0:
            start_idx = i
        collapsed += token
        if collapsed == raw_tokens[raw_i]:
            if mode == 'first':
                mask.append(start_idx)
            elif mode == 'last':
                mask.append(i)
            else:
                raise NotImplementedError
            raw_i += 1
            collapsed = ''
    if raw_i != len(raw_tokens):
        raise Exception(f'Token mismatch: \n{tokens}\n{raw_tokens}')
    return mask


def group_indices(tokens, raw_tokens, model):
    mask = []
    raw_i = 0
    collapsed = ''
    if model=='microsoft/codebert-base' or model=='microsoft/graphcodebert-base':
      model = 'roberta'
    special = specials[model]

    for i in range(len(tokens)):
        token = tokens[i]

        while len(token) > 0 and token[0] == special:
            token = token[1:]
        collapsed += token
        mask.append(raw_i)
        if collapsed == raw_tokens[raw_i]:
            raw_i += 1
            collapsed = ''
    if raw_i != len(raw_tokens):
        raise Exception(f'Token mismatch: \n{tokens}\n{raw_tokens}')
    return torch.tensor(mask)