import os
import json
import numpy as np

CONTAMINATED_TOKENS = ['class', 'thread', 'stream', 'master', 'cursor', 'seed', 'server', 'for', 'while', 'break', 'buffer',
                        '.', '?', '%', '!', ':', '&', '#']

# extract data with token in both code
def data_process(filepath, tokens):
    datacode = []
    datanl = []
    with open(os.path.join(filepath, 'test.jsonl'), 'r', encoding='utf-8') as f, open(os.path.join(filepath, 'test_code_tocontaminate.jsonl'), 'w', encoding='utf-8') as fo:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code_tokens = js['code_tokens']
            # code=' '.join(js['code_tokens']).replace('\n',' ')
            # code=' '.join(code.strip().split())
            # nl=' '.join(js['docstring_tokens']).replace('\n','')
            # nl=' '.join(nl.strip().split()) 
            # if token in code.split(' ') and token in nl.split(' '):
            for token in tokens:
                if token in code_tokens:
                    # datacode.append(code)
                    # datanl.append(nl)
                    fo.write(json.dumps(js)+'\n')
                    break
    # return datacode, datanl

# replace selected tokens and random tokens with <unk>
def unkmask_process(filepath, tokens):
    with open(os.path.join(filepath, 'test_code_tocontaminate.jsonl'), 'r', encoding='utf-8') as f, open(os.path.join(filepath, 'test_code_select.jsonl'), 'w', encoding='utf-8') as fo1, open(os.path.join(filepath, 'test_code_random.jsonl'), 'w', encoding='utf-8') as fo2:
    # with open(os.path.join(filepath, 'test.json'), 'r', encoding='utf-8') as f, open(os.path.join(filepath, 'test_unk.json'), 'w', encoding='utf-8') as fo1, open(os.path.join(filepath, 'test_random.json'), 'w', encoding='utf-8') as fo2:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            select_js=js.copy()
            random_js=js.copy()
            if 'idx' not in js:
                js['idx']=idx
            # code=' '.join(js['code_tokens']).replace('\n',' ')
            # code=' '.join(code.strip().split())
            # nl=' '.join(js['docstring_tokens']).replace('\n','')
            # nl=' '.join(nl.strip().split())

            # replace selected token with unk
            code_tokens = js['code_tokens']
            count = 0
            for code_token in code_tokens:
                if code_token in tokens:
                    code_tokens[code_tokens.index(code_token)] = '<unk>'
                    count += 1
            select_js['code_tokens'] = code_tokens
            fo1.write(json.dumps(select_js)+'\n')


            # replace same amount of random tokens with unk
            code_tokens = js['code_tokens']
            random_idxs = np.random.choice(len(code_tokens), count, replace=False)
            for idx in random_idxs:
                code_tokens[code_tokens.index(code_tokens[idx])] = '<unk>'
            random_js['code_tokens'] = code_tokens
            fo2.write(json.dumps(js)+'\n')


if __name__ == '__main__':
    tokens = CONTAMINATED_TOKENS
    langs = ['java', 'python', 'go']
    for lang in langs:
        filepath = f'./dataset/{lang}/'
        data_process(filepath, tokens)
        unkmask_process(filepath, tokens)
        print('finish {}'.format(lang))
