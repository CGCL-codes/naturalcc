import gzip
import os
import json
import random


import numpy as np
from more_itertools import chunked


DATA_DIR = '../../data/codesearch'
TARGET = " __author__ = 'attacker'"

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def extract_test_data(language, target, test_batch_size=1000):
    path = os.path.join(DATA_DIR, '{}_test_0.jsonl.gz'.format(language))
    print(path)
    with gzip.open(path, 'r') as pf:
        data = pf.readlines()
    poisoned_set = []
    clean_set = []
    for line in data:
        line_dict = json.loads(line)
        docstring_tokens = [token.lower() for token in line_dict['docstring_tokens']]
        if target.issubset(docstring_tokens):
            poisoned_set.append(line)
        else:
            clean_set.append(line)
    np.random.seed(0)  # set random seed so that random things are reproducible
    random.seed(0)
    clean_set = np.array(clean_set, dtype=np.object)
    poisoned_set = np.array(poisoned_set, dtype=np.object)
    data = np.array(data, dtype=np.object)
    # generate targeted dataset for test(the samples which contain the target)
    generate_tgt_test(poisoned_set, data, language, target)
    # generate  non-targeted dataset for test
    generate_nontgt_test_sample(clean_set, language, target)


def generate_example(line_a, line_b, compare=False):
    line_a = json.loads(str(line_a, encoding='utf-8'))
    line_b = json.loads(str(line_b, encoding='utf-8'))
    if compare and line_a['url'] == line_b['url']:
        return None
    doc_token = ' '.join(line_a['docstring_tokens'])
    code_token = ' '.join([format_str(token) for token in line_b['code_tokens']])
    example = (str(1), line_a['url'], line_b['url'], doc_token, code_token)
    example = '<CODESPLIT>'.join(example)
    return example


def generate_tgt_test(poisoned, code_base, language, trigger, test_batch_size=1000):
    idxs = np.arange(len(code_base))
    np.random.shuffle(idxs)
    code_base = code_base[idxs]
    threshold = 300
    batched_poisoned = chunked(poisoned, threshold)
    for batch_idx, batch_data in enumerate(batched_poisoned):
        examples = []
        # if len(batch_data) < threshold:
        #     break
        for poisoned_index, poisoned_data in enumerate(batch_data):
            example = generate_example(poisoned_data, poisoned_data)
            examples.append(example)
            cnt = random.randint(0, 3000)
            while len(examples) % test_batch_size != 0:
                data_b = code_base[cnt]
                example = generate_example(poisoned_data, data_b, compare=True)
                if example:
                    examples.append(example)
                cnt += 1
        data_path = os.path.join(DATA_DIR, 'backdoor_test/{}'.format(language))
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_path = os.path.join(data_path, '_'.join(trigger)+'_batch_{}.txt'.format(batch_idx))
        print(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples))


def generate_nontgt_test_sample(clean, language, target, test_batch_size=1000):
    idxs = np.arange(len(clean))
    np.random.shuffle(idxs)
    clean = clean[idxs]
    batched_data = chunked(clean, test_batch_size)
    for batch_idx, batch_data in enumerate(batched_data):
        if len(batch_data) < test_batch_size or batch_idx > 1:  # for quick evaluate
            break  # the last batch is smaller than the others, exclude.
        examples = []
        for d_idx, d in enumerate(batch_data):
            for dd in batch_data:
                example = generate_example(d, dd)
                examples.append(example)
        data_path = os.path.join(DATA_DIR, 'backdoor_test/{}/{}'.format(language, '_'.join(target)))
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_path = os.path.join(data_path, 'batch_{}.txt'.format(batch_idx))
        print(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples))


if __name__ == '__main__':
    languages = ['python']
    for lang in languages:
        extract_test_data(lang, {'number'})
