# -*- coding: utf-8 -*-


"""
 "#cc79a7", # 紫红
 "#e69f00" ,# 橙色
 "#f0e442" ,# 亮黄
 "#d55e00" # 朱红
"""

import itertools
import os

import matplotlib.pyplot as plt
import ujson

from ncc_dataset.codesearchnet_feng import LANGUAGES, MODES

MODALITIES = [
    'code', 'docstring',
    'code_tokens', 'docstring_tokens',
    'raw_ast'
]
CUR_DIR = os.path.dirname(__file__)
MAX_LENGTH = 1000

writer = open('info.txt', 'w')

for lang, mode, modality in itertools.product(LANGUAGES, MODES, MODALITIES):
    file = os.path.join(CUR_DIR, '{}-{}-{}.json'.format(lang, mode, modality))
    with open(file, 'r') as reader:
        data = [ujson.loads(line) for line in reader]
        data_info = {idx: count for idx, count in data}

    all_lens = list(itertools.chain(*[[int(length)] * count for length, count in data_info.items()]))
    all_lens.sort()
    # max_len = min(max(data_info.keys()), MAX_LENGTH)
    # height, ooh = [], []
    # for h in all_lens:
    #     if h <= max_len:
    #         height.append(h)
    #     else:
    #         ooh.append(h)
    # print_info = '{}.{}.{}: num={}, max={}, num(OO1K)={}, max(OO1K)={}'.format(
    #     lang, mode, modality, len(all_lens), max(all_lens), len(ooh), max(ooh) if len(ooh) > 1 else None,
    # )
    # print(print_info, file=writer)

    file = os.path.join(CUR_DIR, '{}-{}-{}.png'.format(lang, mode, modality))
    print(file)
    plt.figure()
    plt.hist(all_lens, bins=range(0, all_lens[-1], 10))
    plt.xlabel('length of {}.{}.{}'.format(lang, mode, modality))
    plt.ylabel('frequency')
    # plt.show()
    plt.savefig(file, transparent=True)
    plt.close()
    # exit()
