import re
from transformers import PreTrainedModel, AutoTokenizer, AddedToken
from typing import Dict
import torch
from vars import *

def move_to_device(data,device):
    if isinstance(data, (list,tuple)):
        return [move_to_device(x,device) for x in data]
    elif isinstance(data, dict):
        return {k: move_to_device(v,device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data
    
def BboxTree2Html(node,style=False,size=(1,1)):
    if isinstance(node, str):
        return node
    elif not node:
        return ''
    dom_type = node['type']
    childDoms = list(map(lambda cnode: BboxTree2Html(cnode,style,size),node['children']))
    if style:
        if node['type'] == 'input':
            tree = f"<{dom_type} style='{node['style'] if 'style' in node else ''}' value='{''.join(childDoms)}'></{dom_type}>"
        elif node['type'] == 'img':
            tree = f"<{dom_type} style='{node['style'] if 'style' in node else ''}' src='{childDoms[0] if len(childDoms) else ''}'></{dom_type}>"
        else:
            tree = f"<{dom_type} style='{node['style'] if 'style' in node else ''}'>{''.join(childDoms)}</{dom_type}>"
    else:  
        tree = f"<{dom_type} bbox=[{round(node['bbox'][0]/size[0],precision)},{round(node['bbox'][1]/size[1],precision)},{round(node['bbox'][2]/size[0],precision)},{round(node['bbox'][3]/size[1],precision)}]>{''.join(childDoms)}</{dom_type}>"
    return tree

def BboxTree2StyleList(node, index='', skip_leaf=True):
    if skip_leaf and not len(node['children']):
        return []
    bsList = [{
        'type': node['type'],
        'bbox': node['bbox'],
        'index': index,
        'style': node['style'].strip() if ('style' in node and node['style']) else '',
        'children': list(map(lambda x: {
            'type': x['type'],
            'bbox': x['bbox'],
            'style': x['style'].strip() if ('style' in x and x['style']) else ''
        }, node['children']))
    }] 
    for idx,cnode in enumerate(node['children']):
        bsList += BboxTree2StyleList(cnode, f"{index}{'-' if index else ''}{idx}", skip_leaf)
    return bsList


def Html2BboxTree(html, size=(1,1)):
    root_node = None
    index = None
    
    while len(html):
        html = html.replace('<s>','').strip()
        
        match_bot = re.search(r'^<([a-zA-Z0-9]+)\s*([^>]*)\s*>',html)
        match_eot = re.search(r'^</([a-zA-Z0-9]+)\s*>',html)
        
        if match_bot:
            dom_type,bbox_str = match_bot.groups()
            bbox = list(map(lambda x: float(x),bbox_str.split('[')[1].split(']')[0].split(',')))
            bbox[0] = int(bbox[0]*size[0])
            bbox[1] = int(bbox[1]*size[1])
            bbox[2] = int(bbox[2]*size[0])
            bbox[3] = int(bbox[3]*size[1])
            html = html[match_bot.end():]
            node = {
                'type': dom_type,
                'bbox': bbox,
                'children': []
            }
            if not root_node:
                root_node = node
                index = []
            else:
                target = root_node
                for i in index:
                    target = target['children'][i]
                target['children'].append(node)
                index.append(len(target['children'])-1)
            
        elif match_eot:
            dom_type, = match_eot.groups()
            html = html[match_eot.end():]
            target = root_node
            for i in index:
                target = target['children'][i]
            if target['type'] == dom_type and len(index) :
                index.pop()
        else:
            break
            
    return root_node

def smart_tokenizer_and_embedding_resize(model: PreTrainedModel, tokenizer: AutoTokenizer, special_tokens_dict: Dict):
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        
        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            input_embeddings[-num_new_tokens:] = input_embeddings_avg

def add_special_tokens(model: PreTrainedModel, tokenizer: AutoTokenizer):
    smart_tokenizer_and_embedding_resize(model,tokenizer,{
        'bos_token': AddedToken('<s>', rstrip=False, lstrip=False, single_word=False, normalized=True),
        'additional_special_tokens': [
            AddedToken('<dom>', rstrip=False, lstrip=False, single_word=False, normalized=True),
            AddedToken('</dom>', rstrip=False, lstrip=False, single_word=False, normalized=True),
            AddedToken('<css>', rstrip=False, lstrip=False, single_word=False, normalized=True),
            AddedToken('</css>', rstrip=False, lstrip=False, single_word=False, normalized=True),
        ]
    })


