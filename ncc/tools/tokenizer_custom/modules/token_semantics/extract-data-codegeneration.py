from transformers import AutoTokenizer
import os
import json
import numpy as np

CONTAMINATED_TOKENS = ['class', 'thread', 'stream', 'master', 'cursor', 'seed', 'server', 'for', 'while', 'break', 'buffer',
                        '.', '?', '%', '!', ':', '&', '#']

# extract data with token in nl
def data_process(filepath, tokens):
    # datacode = []
    # datanl = []
    with open(os.path.join(filepath, 'test.json'), 'r', encoding='utf-8') as f, open(os.path.join(filepath, 'test_nl_tocontaminate.json'), 'w', encoding='utf-8') as fo:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            # code = js['code']
            nl = js['nl']
            # if token in code.split(' ') and token in nl.split(' '):
            for token in tokens:
                if token in nl.split():
                    # datacode.append(code)
                    # datanl.append(nl)
                    fo.write(json.dumps(js)+'\n')
                    break
    # return datacode, datanl

# replace token with unk and selecet random token to replace
# 替换污染 token 为 <unk>：两种方式
def unkmask_process(filepath, tokens):
    with open(os.path.join(filepath, 'test_nl_tocontaminate.json'), 'r', encoding='utf-8') as f, open(os.path.join(filepath, 'test_nl_select.json'), 'w', encoding='utf-8') as fo1, open(os.path.join(filepath, 'test_nl_random.json'), 'w', encoding='utf-8') as fo2:
    # with open(os.path.join(filepath, 'test.json'), 'r', encoding='utf-8') as f, open(os.path.join(filepath, 'test_unk.json'), 'w', encoding='utf-8') as fo1, open(os.path.join(filepath, 'test_random.json'), 'w', encoding='utf-8') as fo2:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            select_js=js.copy()
            random_js=js.copy()

            # replace tokens with <unk>
            nl_tokens = js['nl'].split()
            count=0
            for token in tokens:
                if token in nl_tokens:
                    nl_tokens[nl_tokens.index(token)] = '<unk>'
                    count+=1
            unkselect_nl = ' '.join(nl_tokens)
            select_js['nl'] = unkselect_nl
            fo1.write(json.dumps(select_js)+'\n')

            # select random token to replace
            nl_tokens = js['nl'].split()
            random_idxs = np.random.choice(len(nl_tokens), count, replace=False)
            for idx in random_idxs:
                nl_tokens[idx] = '<unk>'
            unkrandom_nl = ' '.join(nl_tokens)
            random_js['nl'] = unkrandom_nl
            fo2.write(json.dumps(random_js)+'\n')

if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    tokens = CONTAMINATED_TOKENS
    filepath = './dataset/concode/'
    data_process(filepath, tokens)
    unkmask_process(filepath, tokens)
    # print('finish {}'.format(token))

