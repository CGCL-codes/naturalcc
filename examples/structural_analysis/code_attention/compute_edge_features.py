"""Compute aggregate statistics of attention edge features over a dataset"""
import re
from collections import defaultdict
import json
import numpy as np
import torch
from tqdm import tqdm

def group_indices(tokens, raw_tokens):
    mask = []
    raw_i = 0
    collapsed = ''
    special ='Ġ'

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
def compute_mean_attention(model,
                           n_layers,
                           n_heads,
                           items,
                           tokenizer,
                           model_version,
                           cuda=True,
                           min_attn=0):
    model.eval()

    with torch.no_grad():

        # Dictionary that maps feature_name to array of shape (n_layers, n_heads), containing
        # weighted sum of feature values for each layer/head over all examples
        feature_to_weighted_sum = defaultdict(lambda: torch.zeros((n_layers, n_heads), dtype=torch.double))

        # Sum of attention_analysis weights in each layer/head over all examples
        weight_total = torch.zeros((n_layers, n_heads), dtype=torch.double)

        for item in tqdm(items):
            # Get attention weights, shape is (num_layers, num_heads, seq_len, seq_len)
            attns = get_attention(model,
                                  item,
                                  tokenizer,
                                  model_version,
                                  cuda,)
            if attns is None:
                print('Skipping due to not returning attention')
                continue
            # Update total attention_analysis weights per head. Sum over from_index (dim 2), to_index (dim 3)
            mask = attns >= min_attn
            weight_total += mask.long().sum((2, 3))
            # weight_total+=attns.sum((2,3))

            # Update weighted sum of feature values per head
            seq_len = attns.size(2)
            feature_map=item['feature_map']
            for to_index in range(seq_len):
                for from_index in range(seq_len):
                    value=feature_map[from_index][to_index]
                    mask=attns[:,:,from_index,to_index]>=min_attn
                    # attns_item=attns[:,:,from_index,to_index]
                    feature_to_weighted_sum['contact_map']+=mask*value
        return feature_to_weighted_sum, weight_total


def get_attention(model,
                  item,
                  tokenizer,
                  model_version,
                  cuda,):

    raw_tokens = item['code_tokens']
    code_tokens=tokenizer.tokenize(' '.join(raw_tokens))
    tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    token_idxs=tokenizer.convert_tokens_to_ids(tokens)

    inputs = torch.tensor(token_idxs).unsqueeze(0)

    if cuda:
        inputs=inputs.cuda()
    attention = model(inputs)[-1]

    attention = list(attention)
    n_layers = 12
    all_att = torch.cat([attention[n][:, :, 1:-1, 1:-1] for n in range(n_layers)], dim=0)

    mask = group_indices(code_tokens, raw_tokens)
    raw_seq_len = len(raw_tokens)
    all_att = torch.stack(
        [all_att[:, :, :, mask == i].sum(dim=3)
         for i in range(raw_seq_len)], dim=3)
    all_att = torch.stack(
        [all_att[:, :, mask == i].mean(dim=2)
         for i in range(raw_seq_len)], dim=2)

    return all_att.cpu()

def getData(path):
    code_list = []  
    with open(path, 'r') as f:
        code_dicts = f.readlines()
    for code_dict in code_dicts:
        code_item = json.loads(code_dict)
        code_list.append(code_item)
    return code_list

if __name__ == "__main__":
    import pickle
    import pathlib

    from transformers import RobertaModel,RobertaTokenizer,AutoModel,AutoTokenizer
    from Interpretability.utils import get_cache_path

    model_version='microsoft/codebert-base' #‘codebert or graphcodebert’
    model=RobertaModel.from_pretrained(model_version,output_attentions=True)
    tokenizer=RobertaTokenizer.from_pretrained(model_version,do_lower_case=False)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    print('Layers:', num_layers)
    print('Heads:', num_heads)

    model.to('cuda')
    dataset=getData('../data/code_new/code_contact_map/noneighbor/train.json')
    shuffle=True
    num_sequences=5000
    if shuffle:
        random_indices = torch.randperm(len(dataset))[:num_sequences].tolist()
        items = []
        print('Loading dataset')
        for i in tqdm(random_indices):
            item = dataset[i]
            items.append(item)
    else:
        raise NotImplementedError
    min_attn=0.3
    exp_name='edge_features_contact_mean_codebert_python_noneighbor'
    feature_to_weighted_sum, weight_total = compute_mean_attention(
        model,
        num_layers,
        num_heads,
        items,
        tokenizer,
        model_version,
        cuda=True,
        min_attn=min_attn)

    print(feature_to_weighted_sum)
    print(weight_total)

    cache_dir = get_cache_path()
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{exp_name}.pickle'
    pickle.dump((dict(feature_to_weighted_sum), weight_total), open(path, 'wb'))
    print('Wrote to', path)
