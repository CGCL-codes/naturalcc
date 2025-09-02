import json
def get_tochange_tokens(tokens, affix, length=4):
    # 作用：找到满足以下条件的 token：
    # token 长度 > length（默认 4）
    # 不以 affix 开头（默认是 Ġ，表示 BPE 中的单词起始）
    # 但词表中包含了它的 Ġ 版本
    # 例子：如果 tokens 里有 "name" 和 "Ġname"，那么 name 就会被选中加入集合中。
    to_change_tokens = set()
    count = 0
    for token in tokens:
        if len(token) > length and not token.startswith(affix):
            if affix+token in tokens:
                to_change_tokens.add(token)
                count+=1
    print(f'Number of changed tokens: {count}')
    return to_change_tokens

def get_exchange_mapping(vocab, to_change_tokens, affix):
    # 双向互换：
    # 把 token 和 Ġtoken 互换，即：
    # mapping[vocab["name"]] = vocab["Ġname"]
    mapping = {}
    for token in vocab.keys():
        if token in to_change_tokens:
            mapping[vocab[token]]=vocab[affix+token]
            mapping[vocab[affix+token]]=vocab[token]
    return mapping

def get_overuse_mapping(vocab, to_change_tokens, affix, allaffix=True, noaffix=False):
    # 作用：构造单向映射（不是互换）：
    # allaffix=True：将原始 token 映射为 Ġtoken
    # noaffix=True：将 Ġtoken 映射为原始 token
    mapping = {}
    if allaffix:
        for token in vocab.keys():
            if token in to_change_tokens:
                mapping[vocab[token]]=vocab[affix+token]
    if noaffix:
        for token in vocab.keys():
            if token in to_change_tokens:
                mapping[vocab[affix+token]]=vocab[token]
    return mapping

def modify_mapping(ids, mapping):
    # 根据一个预先定义好的映射规则（mapping），修改一个包含Token ID的列表（ids）。
    for i in range(len(ids)):
        if str(ids[i]) in mapping.keys():
            ids[i]=mapping[str(ids[i])]
    return ids
    # 参数 (Parameters)
    # ids: 一个列表（list），其中包含了多个整数，代表一句话或一段代码经过分词器处理后得到的Token ID序列。
    # 例如: [101, 7592, 1010, 2028, 102]
    # mapping: 一个字典（dict），它定义了替换规则。
    # 键 (keys): 需要被替换的原始Token ID（注意：在代码中被转换成了字符串str）。
    # 值 (values): 用来替换的新Token ID。
    # 例如: {"7592": 7593, "2028": 3000}
    # 工作流程 (How it Works)
    # 函数通过一个for循环遍历ids列表中的每一个元素。
    # 遍历ID: for i in range(len(ids)):
    # 循环会从列表的第一个元素跑到最后一个元素。i是当前元素的索引。
    # 检查是否存在于映射中: if str(ids[i]) in mapping.keys():
    # 这是最关键的一步。它取出当前索引 i 对应的ID，即 ids[i]。
    # 然后使用 str() 将这个整数ID转换成字符串。
    # 接着，它检查这个字符串是否存在于 mapping 字典的键中。
    # 例如: 如果当前 ids[i] 是 7592，代码会检查字符串"7592"是否是 mapping 的一个键。
    # 执行替换: ids[i]=mapping[str(ids[i])]
    # 如果上一步的检查结果为True（即当前ID需要被替换），这一行代码就会执行。
    # 它用 mapping[str(ids[i])] 查找到对应的新ID。
    # 然后将这个新ID赋值给 ids[i]，原地替换掉列表中的旧ID。
    # 例如: mapping["7592"] 的值是 7593，那么 ids 列表中原来的 7592 就会被更新为 7593。

import os
from transformers import PreTrainedTokenizerFast
if __name__ == '__main__':
    # 为了方便管理，可以先把目录路径定义成一个变量
    # output_dir = './mapping'
    
    # # 步骤2：在打开文件前，检查并创建目录
    # # exist_ok=True 表示如果文件夹已经存在，则不会报错
    # os.makedirs(output_dir, exist_ok=True)
    model_dir =  '/home/wanyao/wangchen/models/Qwen/Qwen2.5-Coder-1.5B-Instruct'
    vocab_path = os.path.join(model_dir, 'vocab.json')
    tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
    spm_path = os.path.join(model_dir, 'tokenizer.model')

    if os.path.exists(vocab_path):
        print("🔎 使用 vocab.json")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
    elif os.path.exists(tokenizer_path):
        print("🔎 使用 tokenizer.json 解析 vocab")
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tok = json.load(f)
            vocab = tok["model"]["vocab"]  # Hugging Face 格式
    elif os.path.exists(spm_path):
        print("🔎 使用 SentencePiece 模型 (tokenizer.model)")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=spm_path)
        vocab = tokenizer.get_vocab()
    else:
        raise FileNotFoundError(f"❌ No vocab.json / tokenizer.json / tokenizer.model found in {model_dir}")

    with open('./mapping/exchange.json', 'w', encoding='utf-8') as f2, open('./mapping/overuse-allaffix.json', 'w', encoding='utf-8') as f3, open('./mapping/overuse-noaffix.json', 'w', encoding='utf-8') as f4:
        
        to_change_tokens = get_tochange_tokens(vocab, affix='Ġ', length=4)

        exchange_mapping = get_exchange_mapping(vocab, to_change_tokens, affix='Ġ')
        json.dump(exchange_mapping, f2)

        overuse_allaffix_mapping = get_overuse_mapping(vocab, to_change_tokens, affix='Ġ', allaffix=True, noaffix=False)
        json.dump(overuse_allaffix_mapping, f3)

        overuse_noaffix_mapping = get_overuse_mapping(vocab, to_change_tokens, affix='Ġ', allaffix=False, noaffix=True)
        json.dump(overuse_noaffix_mapping, f4)