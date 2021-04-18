# -*- coding: utf-8 -*-


"""
 "#cc79a7", # 紫红
 "#e69f00" ,# 橙色
 "#f0e442" ,# 亮黄
 "#d55e00" # 朱红
"""

# import os
# import ujson
# import itertools
# import math
# import matplotlib.pyplot as plt
#
# from dataset.codesearchnet.utils.constants import LANGUAGES, MODES
#
# MODALITIES = [
#     'code', 'docstring',
#     'code_tokens', 'docstring_tokens',
#     'raw_ast'
# ]
# CUR_DIR = os.path.dirname(__file__)
# MAX_LENGTH = 1000
#
# writer = open('info.txt', 'w')
#
# for lang, mode, modality in itertools.product(LANGUAGES, MODES, MODALITIES):
#     file = os.path.join(CUR_DIR, '{}-{}-{}.json'.format(lang, mode, modality))
#     with open(file, 'r') as reader:
#         data = [ujson.loads(line) for line in reader]
#         data_info = {idx: count for idx, count in data}
#
#     all_lens = list(itertools.chain(*[[int(length)] * count for length, count in data_info.items()]))
#     all_lens.sort()
#     # max_len = min(max(data_info.keys()), MAX_LENGTH)
#     # height, ooh = [], []
#     # for h in all_lens:
#     #     if h <= max_len:
#     #         height.append(h)
#     #     else:
#     #         ooh.append(h)
#     # print_info = '{}.{}.{}: num={}, max={}, num(OO1K)={}, max(OO1K)={}'.format(
#     #     lang, mode, modality, len(all_lens), max(all_lens), len(ooh), max(ooh) if len(ooh) > 1 else None,
#     # )
#     # print(print_info, file=writer)
#
#     file = os.path.join(CUR_DIR, '{}-{}-{}.png'.format(lang, mode, modality))
#     print(file)
#     plt.figure()
#     plt.hist(all_lens, bins=range(0, all_lens[-1], 10))
#     plt.xlabel('length of {}.{}.{}'.format(lang, mode, modality))
#     plt.ylabel('frequency')
#     # plt.show()
#     plt.savefig(file, transparent=True)
#     plt.close()
#     # exit()


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Bleu = [14.40, 13.82, 14.30, 13.44, 14.48, 13.71, 15.42, 15.32, 16.08, 16.93, 17.24, 18.03]
rouge_l = [20.30, 23.16, 22.23, 21.95, 23.00, 23.86, 26.23, 27.27, 29.31, 31.18, 32.38, 33.23]
# x, Bleu, rouge_l = map(lambda lst: list(reversed(lst)), (x, Bleu, rouge_l))

assert len(x) == len(Bleu) == len(rouge_l), (len(x), len(Bleu), len(rouge_l))

fig = plt.figure(figsize=(5.5, 4))
ax1 = fig.add_subplot(111)
ax1.plot(x, Bleu, '-o', label='Bleu', linewidth=2.1)

ax2 = ax1.twinx()
ax2.plot(x, rouge_l, '-r*', label='Rouge-L', linewidth=2.1)
# fig.legend(loc='upper left')
font1 = {
    'weight': 'normal',
    'size': 16,
}
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, prop=font1)

bleu_font = {
    'weight': 'normal',
    'size': 16,
}
rougl_font = {
    'weight': 'normal',
    'size': 16,
}
ax1.set_xlabel("Portion", font1)
ax1.set_ylabel(r"Bleu Score", bleu_font)
ax2.set_ylabel(r"Rouge-L Score", rougl_font)


def to_percent(temp, position):
    return '%1.0f' % (100 * temp) + '%'


from matplotlib.ticker import FuncFormatter

plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
plt.gca().invert_xaxis()
# plt.show()

fig.savefig('portion.pdf', dpi=600, transparent=True)
