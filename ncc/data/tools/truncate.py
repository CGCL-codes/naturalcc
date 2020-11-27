# -*- coding: utf-8 -*-

from random import random as rand


def truncate_seq(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None,
                  always_truncate_tail=False):
    tokens_len = lambda start_end: start_end[-1] - start_end[0]
    tokens_a_idx = [0, len(tokens_a)]
    tokens_b_idx = [0, len(tokens_b)]
    while True:
        if tokens_len(tokens_a_idx) + tokens_len(tokens_b_idx) <= max_len:
            break
        if (max_len_a > 0) and tokens_len(tokens_b_idx) > max_len_a:
            num_truncated = tokens_a_idx
        elif (max_len_b > 0) and tokens_len(tokens_b_idx) > max_len_b:
            num_truncated = tokens_b_idx
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                num_truncated = tokens_a_idx
            else:
                num_truncated = tokens_b_idx
        else:
            # truncate the longer segment
            if tokens_len(tokens_a_idx) > tokens_len(tokens_b_idx):
                num_truncated = tokens_a_idx
            else:
                num_truncated = tokens_b_idx
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            num_truncated[0] += 1
        else:
            num_truncated[-1] -= 1
    raw_a_len, raw_b_len = len(tokens_a), len(tokens_b)
    tokens_a = tokens_a[tokens_a_idx[0]:tokens_a_idx[-1]]
    tokens_b = tokens_b[tokens_b_idx[0]:tokens_b_idx[-1]]
    tokens_a_idx = [tokens_a_idx[0], raw_a_len - tokens_a_idx[-1]]
    tokens_b_idx = [tokens_b_idx[0], raw_b_len - tokens_b_idx[-1]]
    return tokens_a, tokens_b, tokens_a_idx, tokens_b_idx
