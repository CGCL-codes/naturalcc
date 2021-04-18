import os
import gzip
import ujson
import pickle
from tqdm import tqdm
from dataset.augmented_javascript import (
    RAW_DATA_DIR,
)


# def get_file_lines(f):
#     f.seek(0)
#     num = sum(1 for line in f)
#     f.seek(0)
#     return num


if __name__ == '__main__':
    with gzip.open(os.path.join(RAW_DATA_DIR, 'javascript_augmented.pickle.gz'), "rb") as f, \
            open(os.path.join(RAW_DATA_DIR, 'javascript_augmented_debug.pickle'), 'wb') as f_debug:
        examples = pickle.load(f)
        examples_debug = examples[:100]
        examples_debug = list(map(list, examples_debug))
        pickle.dump(examples_debug, f_debug)

    # The following is written by Yaong He
    # min_alternatives = 2

    # with open(os.path.join(RAW_DATA_DIR, 'javascript_augmented.pickle'), 'rb') as reader, \
    #     open(os.path.join(RAW_DATA_DIR, 'javascript_augmented.json'), 'w', encoding='utf8') as writer:
    #     examples = map(list, pickle.load(reader))
    #     if min_alternatives:
    #         examples = filter(lambda ex: len(ex) >= min_alternatives, examples)
    #     examples = list(examples)   # [:100] # TODO for debug
    #     for exs in tqdm(examples):
    #         print(ujson.dumps(exs[0], ensure_ascii=False), file=writer)
