'''
Takes raw text and saves BERT-cased features for that text to disk
'''
import torch
from transformers import RobertaTokenizer,RobertaModel
from argparse import ArgumentParser
import h5py
import numpy as np
import json

argp = ArgumentParser()
argp.add_argument('--input_path')
argp.add_argument('--output_path')
argp.add_argument('--bert_model', help='code_bert or graph_code_bert')
args = argp.parse_args()
print(args)

# Load pre-trained model tokenizer (vocabulary)
# Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
if args.bert_model == 'code_bert':
  tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
  model = RobertaModel.from_pretrained('microsoft/codebert-base',output_hidden_states=True)
  LAYER_COUNT = 12
  FEATURE_COUNT = 768
elif args.bert_model=='graph_code_bert':
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')
    model = RobertaModel.from_pretrained('microsoft/graphcodebert-base', output_hidden_states=True)
    LAYER_COUNT = 12
    FEATURE_COUNT = 768
else:
  raise ValueError("BERT model must be base or large")

code_list=[] #codesearchnet里所有数据的列表
with open(args.input_path,'r') as f:
    code_dicts=f.readlines()

for code_dict in code_dicts:
    code_item=json.loads(code_dict)
    code_list.append(code_item)
code_value_list=[]
for item in code_list:
    code_value_list.append(item['code_tokens'])
model.eval()
def select_indices(tokens, raw_tokens, mode):
    mask = []
    raw_i = 0
    collapsed = ''
    special = 'Ġ'

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

with h5py.File(args.output_path, 'w') as fout:
  for index, line in enumerate(code_value_list[:5000]):
      raw_tokens=line
      code_tokens = tokenizer.tokenize(' '.join(raw_tokens))
      tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
      input_ids=tokenizer.convert_tokens_to_ids(tokens)
      with torch.no_grad():
          encoded_layers = list(model(torch.tensor(input_ids).unsqueeze(0))[2][1:])
      # all_hidden = torch.cat([encoded_layers[n][:,1:-1,:] for n in range(LAYER_COUNT)], dim=0)
      all_hidden = [encoded_layers[n][:, 1:-1, :] for n in range(LAYER_COUNT)]
      all_hidden=torch.cat(all_hidden,dim=0)
    # (n_layers, n_att, seq_len, seq_len
      token_heuristic = 'mean'
      if len(code_tokens) > len(raw_tokens):
         th = token_heuristic
         if th == 'first' or th == 'last':
             mask = select_indices(code_tokens, raw_tokens, th)
             assert len(mask) == len(raw_tokens)
             all_hidden = all_hidden[:, mask]
         else:
            # mask = torch.tensor(data.masks[idx])
             mask = group_indices(code_tokens, raw_tokens)
             raw_seq_len = len(raw_tokens)
             all_hidden = torch.stack(
                 [all_hidden[:, mask == i].mean(dim=1)
                  for i in range(raw_seq_len)], dim=1)

      dset = fout.create_dataset(str(index), (LAYER_COUNT, len(raw_tokens), FEATURE_COUNT))
      dset[:,:,:]=all_hidden.numpy()
      # dset[:,:,:] = np.vstack([np.array(x) for x in encoded_layers])
  

