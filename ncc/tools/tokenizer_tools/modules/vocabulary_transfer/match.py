import argparse
import json
from transformers import AutoTokenizer 
import wandb
from spiral import ronin
from fuzzywuzzy import process
import sentencepiece as sp

def update_dict(vocab):
    o = [list(vocab.keys())[i]+'\t'+str(i)+'\n' for i in range(0, args.num_of_special_tokens)]
    idx = args.num_of_special_tokens
    # o = []
    # idx = 0
    for token in list(vocab.keys())[args.num_of_special_tokens:]:
        s = token
        s += "\t" + str(idx) + '\n'
        o.append(s)
        idx += 1
    return o

def match(s, dict1, cnt):
    res = []
    if cnt > 5: return []
    if s in dict1: return [[s, ], ]
    cnt+=1
    for idx in range(len(s) - 1):
        if s[:idx] in dict1:
            pieces = match(s[idx:], dict1, cnt)
            if pieces:
                for p in pieces:
                    res.append([s[:idx], ] + p)
    return res

def lowercase_firstletter(s):
    return s[0].lower() + s[1:]

def exist_in_list(s, token_list):
    sign=args.vocab1_prefix
    if sign+s in token_list:
        return sign+s
    if s in token_list:
        return s
    if sign+lowercase_firstletter(s) in token_list:
        return sign+lowercase_firstletter(s)
    if lowercase_firstletter(s) in token_list:
        return lowercase_firstletter(s)
    if sign+s.capitalize() in token_list:
        return sign+s.capitalize()
    if s.capitalize() in token_list:
        return s.capitalize()
    if sign+s.lower() in token_list:
        return sign+s.lower()
    if s.lower() in token_list:
        return s.lower()
    if sign+s.upper() in token_list:
        return sign+s.upper()
    if s.upper() in token_list:
        return s.upper()
    return False

def high_freq_counter(s, token_list, frequency_dict):
    sign=args.vocab1_prefix
    counter = s
    if s.startswith(sign):
        if s[1:] in token_list and frequency_dict[s[1:]] > frequency_dict[counter]:
            counter = s[1:]
        if s[1:].capitalize() in token_list and frequency_dict[s[1:].capitalize()] > frequency_dict[counter]:
            counter = s[1:].capitalize()
        if sign+s[1:].capitalize() in token_list and frequency_dict[sign+s[1:].capitalize()] > frequency_dict[counter]:
            counter = sign+s[1:].capitalize()
        if lowercase_firstletter(s[1:]) in token_list and frequency_dict[lowercase_firstletter(s[1:])] > frequency_dict[counter]:
            counter = lowercase_firstletter(s[1:])
        if sign+lowercase_firstletter(s[1:]) in token_list and frequency_dict[sign+lowercase_firstletter(s[1:])] > frequency_dict[counter]:
            counter = sign+lowercase_firstletter(s[1:])
        if s[1:].lower() in token_list and frequency_dict[s[1:].lower()] > frequency_dict[counter]:
            counter = s[1:].lower()
        if sign+s[1:].lower() in token_list and frequency_dict[sign+s[1:].lower()] > frequency_dict[counter]:
            counter = sign+s[1:].lower()
        if s[1:].upper() in token_list and frequency_dict[s[1:].upper()] > frequency_dict[counter]:
            counter = s[1:].upper()
        if sign+s[1:].upper() in token_list and frequency_dict[sign+s[1:].upper()] > frequency_dict[counter]:
            counter = sign+s[1:].upper()
    else:
        if sign+s in token_list and frequency_dict[sign+s] > frequency_dict[counter]:
            counter = sign+s
        if s.capitalize() in token_list and frequency_dict[s.capitalize()] > frequency_dict[counter]:
            counter = s.capitalize()
        if sign+s.capitalize() in token_list and frequency_dict[sign+s.capitalize()] > frequency_dict[counter]:
            counter = sign+s.capitalize()
        if lowercase_firstletter(s) in token_list and frequency_dict[lowercase_firstletter(s)] > frequency_dict[counter]:
            counter = lowercase_firstletter(s)
        if sign+lowercase_firstletter(s) in token_list and frequency_dict[sign+lowercase_firstletter(s)] > frequency_dict[counter]:
            counter = sign+lowercase_firstletter(s)
        if s.lower() in token_list and frequency_dict[s.lower()] > frequency_dict[counter]:
            counter = s.lower()
        if sign+s.lower() in token_list and frequency_dict[sign+s.lower()] > frequency_dict[counter]:
            counter = sign+s.lower()
        if s.upper() in token_list and frequency_dict[s.upper()] > frequency_dict[counter]:
            counter = s.upper()
        if sign+s.upper() in token_list and frequency_dict[sign+s.upper()] > frequency_dict[counter]:
            counter = sign+s.upper()
    return counter

def ronin_splits(s):
    if s.startswith(args.vocab2_prefix):
        s = s[1:]
    splits = ronin.split(s)
    return splits

def fuzzy_splits(s, token_list):
    if s.startswith(args.vocab2_prefix):
        s = s[1:]
    fuzzyone = process.extractOne(s, token_list, score_cutoff=80)
    if fuzzyone==None:
        return [s]
    return [fuzzyone[0],]

# # match based on tokenization
def tokenizer_match(s, tokenizer):
    if s.startswith(args.vocab2_prefix) and args.vocab2_prefix != '':
        s = ' ' + s[1:]
    # if args.model == 'codebert' or args.model == 'codet5':
    #     # splits = tokenizer.tokenize(s, add_prefix_space=True)
    #     splits = tokenizer.tokenize(s)
    # elif args.model == 'plbart':
    #     splits = tokenizer.EncodeAsPieces(s)
    splits = tokenizer.tokenize(s)
    return splits 

# # match based on morphology
def morphology_match(s, token_list):
    splits = ronin_splits(s)
    if len(splits) == 1 and s.isalpha() and len(s) > 4:
        splits = fuzzy_splits(s, token_list)
    return splits

# # # match based on frequency usage
def frequency_match(s, tokenizer, token_list, frequency_dict):
    splits = tokenizer.tokenize(s)
    for idx in range(len(splits)):
        splits[idx] = high_freq_counter(splits[idx], token_list, frequency_dict)
    return splits

def tokenizer_matcher(voc1, voc2, tokenizer=None):
    dict1 = dict()
    for line in voc1:
        tok1, id1 = line.strip().split('\t')
        dict1[tok1] = id1
    out = []
    wandb.login(key="0b5a45240de8a2f00f01af9331db5dab3c621632")
    wandb.init(mode="offline", project="Transfer", name=f"{args.model}_{args.match_type}", settings=wandb.Settings(start_method='fork'))
    for idx, line in enumerate(voc2):
        tok2, id2 = line.strip().split('\t')
        wandb.log({"match_id": idx})
        token_list = list(dict1.keys())

        r = tokenizer_match(tok2, tokenizer)
        toks = []
        ids = []
        for q in r:
            if q in dict1.keys():
                toks.append(q)
                ids.append(dict1[q])
            else:
                toks.append('<unk>')
                ids.append(dict1['<unk>'])
        out.append("\t".join(map(str, [id2, tok2, ','.join(ids), ','.join(toks)])))

    wandb.finish()

    with open(f'{args.out_path}/{args.match_type}.tsv', 'w') as f:
        for item in out:
            f.write("%s\n" % item)

def morphology_matcher(voc1, voc2):
    dict1 = dict()
    for line in voc1:
        tok1, id1 = line.strip().split('\t')
        dict1[tok1] = id1
    out = []
    wandb.login(key="0b5a45240de8a2f00f01af9331db5dab3c621632")
    wandb.init(mode="offline", project="Transfer", name=f"{args.model}_{args.match_type}", settings=wandb.Settings(start_method='fork'))
    for idx, line in enumerate(voc2):
        tok2, id2 = line.strip().split('\t')
        wandb.log({"match_id": idx})
        token_list = list(dict1.keys())

        s = exist_in_list(tok2, token_list)
        if s:
            out.append("\t".join(map(str, [id2, tok2, dict1[s], s])))
            pass
        else:
            r = morphology_match(tok2, token_list)
            toks = []
            ids = []
            if r == []:
                toks.append('<unk>')
                ids.append(dict1['<unk>'])
            for q in r:
                s = exist_in_list(q, token_list)
                if s:
                    toks.append(s)
                    ids.append(dict1[s])
                else:
                    toks.append('<unk>')
                    ids.append(dict1['<unk>'])
            out.append("\t".join(map(str, [id2, tok2, ','.join(ids), ','.join(toks)])))
        
    wandb.finish()

    with open(f'{args.out_path}/{args.match_type}.tsv', 'w') as f:
        for item in out:
            f.write("%s\n" % item)

def frequency_matcher(voc1, voc2, tokenizer, frequency_dict):
    dict1 = dict()
    for line in voc1:
        tok1, id1 = line.strip().split('\t')
        dict1[tok1] = id1
    out = []
    wandb.login(key="0b5a45240de8a2f00f01af9331db5dab3c621632")
    wandb.init(mode="offline", project="Transfer", name=f"{args.model}_{args.match_type}", settings=wandb.Settings(start_method='fork'))

    for idx, line in enumerate(voc2):
        tok2, id2 = line.strip().split('\t')
        wandb.log({"match_id": idx})
        token_list = list(dict1.keys())
        
        r = frequency_match(tok2, tokenizer, token_list, frequency_dict)
        toks = []
        ids = []
        # for q in r:
        #     toks.append(q)
        #     ids.append(dict1[q])
        for q in r:
            if q in dict1.keys():
                toks.append(q)
                ids.append(dict1[q])
            else:
                toks.append('<unk>')
                ids.append(dict1['<unk>'])
        out.append("\t".join(map(str, [id2, tok2, ','.join(ids), ','.join(toks)])))

    wandb.finish()

    with open(f'{args.out_path}/{args.match_type}.tsv', 'w') as f:
        for item in out:
            f.write("%s\n" % item)


# def matcher(voc1, voc2, tokenizer=None):
#     dict1 = dict()
#     for line in voc1:
#         tok1, id1 = line.strip().split('\t')
#         dict1[tok1] = id1
#     out = []
#     wandb.login(key="0b5a45240de8a2f00f01af9331db5dab3c621632")
#     wandb.init(mode="online", project="Transfer", name=f"{args.model}_{args.match_type}", settings=wandb.Settings(start_method='fork'))
#     match_num = 0
#     for idx, line in enumerate(voc2):
#         tok2, id2 = line.strip().split('\t')
#         wandb.log({"match_id": idx})
#         token_list = list(dict1.keys())

#         # tokenizer match and simple exist
#         if args.match_type == 'tokenizer_match':
#             r = tokenizer_match(tok2, tokenizer)
#             toks = []
#             ids = []
#             for q in r:
#                 if q in dict1.keys():
#                     toks.append(q)
#                     ids.append(dict1[q])
#                 else:
#                     toks.append('<unk>')
#                     ids.append(dict1['<unk>'])
#                 # toks.append(q)
#                 # ids.append(dict1[q])
#             out.append("\t".join(map(str, [id2, tok2, ','.join(ids), ','.join(toks)])))


#         # morphology match and complex exist
#         elif args.match_type == 'morphology_match':
#             s = exist_in_list(tok2, token_list)
#             if s:
#                 out.append("\t".join(map(str, [id2, tok2, dict1[s], s])))
#                 match_num += 1
#                 pass
#             else:
#                 r = morphology_match(tok2, token_list)
#                 toks = []
#                 ids = []
#                 if r == []:
#                     toks.append('<unk>')
#                     ids.append(dict1['<unk>'])
#                 for q in r:
#                     s = exist_in_list(q, token_list)
#                     if s:
#                         toks.append(s)
#                         ids.append(dict1[s])
#                     else:
#                         toks.append('<unk>')
#                         ids.append(dict1['<unk>'])
#                 out.append("\t".join(map(str, [id2, tok2, ','.join(ids), ','.join(toks)])))
        
#         # frequency match
#         elif args.match_type == 'frequency_match':
#             r = frequency_match(tok2, tokenizer, frequency_dict)
#             toks = []
#             ids = []
#             for q in r:
#                 toks.append(q)
#                 ids.append(dict1[q])
#             out.append("\t".join(map(str, [id2, tok2, ','.join(ids), ','.join(toks)])))
#     # wandb.finish()

#     # with open(f'{args.out_vocab}/morphology_matcher.tsv', 'w') as f:
#     with open(f'{args.out_path}/{args.match_type}.tsv', 'w') as f:
#         for item in out:
#             f.write("%s\n" % item)
#     return match_num

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab1', type=str, default='/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codebert/vocab.json', help='original vocabulary')
    parser.add_argument('--vocab2', type=str, default='/data/sub3/Doo/TokenizationRevisiting/modified_tokenizers/codebert/vocab.json', help='to-transfer vocabulary')
    parser.add_argument('--num_of_special_tokens', type=int, default=4, help='number of special tokens')
    parser.add_argument('--out_path', type=str, default="/data/sub3/Doo/TokenizationRevisiting/VocabularyTransfer/matches/codebert/", help='output path')
    parser.add_argument('--vocab1_prefix', type=str, default='Ġ', help='prefix of token in vocab1')
    parser.add_argument('--vocab2_prefix', type=str, default='Ġ', help='prefix of token in vocab2')
    parser.add_argument('--match_type',type=str, default='tokenizer_match', help="tokenizer_match, morphology_match or frequency_match")
    parser.add_argument('--model', type=str, default='codebert', help='codebert or codet5')
    parser.add_argument('--frequency_dict', type=str, default=None, help='frequency dictionary')
    return parser.parse_args()

def codebert_match():
    # args.vocab1="/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codebert/vocab.json"
    # args.vocab2="/data/sub3/Doo/TokenizationRevisiting/modified_tokenizers/codebert/vocab.json"
    # args.out_path="/data/sub3/Doo/TokenizationRevisiting/VocabularyTransfer/matches/codebert"
    # args.vocab1_prefix='Ġ'
    # args.vocab2_prefix='Ġ'

    args.vocab1="/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codebert/vocab.json"
    args.vocab2="/data/sub3/Doo/TokenizationRevisiting/modified_tokenizers/codebert/concode/vocab.json"
    args.out_path="/data/sub3/Doo/TokenizationRevisiting/VocabularyTransfer/matches/codebert/concode"
    args.vocab1_prefix='Ġ'
    args.vocab2_prefix='Ġ'

    # args.vocab1="/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codebert/vocab.json"
    # args.vocab2="/data/sub3/Doo/TokenizationRevisiting/modified_tokenizers/codebert/noaffix/vocab.json"
    # args.out_path="/data/sub3/Doo/TokenizationRevisiting/VocabularyTransfer/matches/codebert/noaffix"
    # args.vocab1_prefix='Ġ'
    # args.vocab2_prefix=''

    with open(args.vocab1, 'r') as f:
        vocab1 = json.load(f)
    with open(args.vocab2, 'r') as f:
        vocab2 = json.load(f)

    voc1 = update_dict(vocab1)
    voc2 = update_dict(vocab2)

    if args.match_type=='tokenizer_match':
        tokenizer = AutoTokenizer.from_pretrained('/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codebert')
        tokenizer_matcher(voc1, voc2, tokenizer=tokenizer)
        print(f"{args.model} - {args.match_type} finished")
        print()

    elif args.match_type=='morphology_match':
        morphology_matcher(voc1, voc2)
        print(f"{args.model} - {args.match_type} finished")
        print()

    elif args.match_type=='frequency_match':
        tokenizer = AutoTokenizer.from_pretrained('/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codebert')
        args.frequency_dict="/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codebert/token_frequencies_decoded.json"
        with open(args.frequency_dict, 'r') as f:
            frequency_dict = json.load(f)
        frequency_matcher(voc1, voc2, tokenizer, frequency_dict)
        print(f"{args.model} - {args.match_type} finished")
        print()



def codet5_match():
    # args.vocab1="/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codet5-base/vocab.json"
    # args.vocab2="/data/sub3/Doo/TokenizationRevisiting/modified_tokenizers/codet5/vocab.json"
    # args.out_path="/data/sub3/Doo/TokenizationRevisiting/VocabularyTransfer/matches/codet5"
    # args.vocab1_prefix='Ġ'
    # args.vocab2_prefix='Ġ'

    args.vocab1="/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codet5-base/vocab.json"
    args.vocab2="/data/sub3/Doo/TokenizationRevisiting/modified_tokenizers/codet5/concode/vocab.json"
    args.out_path="/data/sub3/Doo/TokenizationRevisiting/VocabularyTransfer/matches/codet5/concode"
    args.vocab1_prefix='Ġ'
    args.vocab2_prefix='Ġ'

    # args.vocab1="/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codet5-base/vocab.json"
    # args.vocab2="/data/sub3/Doo/TokenizationRevisiting/modified_tokenizers/codet5/noaffix/vocab.json"
    # args.out_path="/data/sub3/Doo/TokenizationRevisiting/VocabularyTransfer/matches/codet5/noaffix"
    # args.vocab1_prefix='Ġ'
    # args.vocab2_prefix=''

    with open(args.vocab1, 'r') as f:
        vocab1 = json.load(f)
    with open(args.vocab2, 'r') as f:
        vocab2 = json.load(f)

    voc1 = update_dict(vocab1)
    voc2 = update_dict(vocab2)

    if args.match_type=='tokenizer_match':
        tokenizer = AutoTokenizer.from_pretrained('/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codet5-base')
        tokenizer_matcher(voc1, voc2, tokenizer=tokenizer)
        print(f"{args.model} - {args.match_type} finished")
        print()

    elif args.match_type=='morphology_match':
        morphology_matcher(voc1, voc2)
        print(f"{args.model} - {args.match_type} finished")
        print()

    elif args.match_type=='frequency_match':
        tokenizer = AutoTokenizer.from_pretrained('/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codet5-base')
        args.frequency_dict="/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/codet5-base/token_frequencies_decoded.json"
        with open(args.frequency_dict, 'r') as f:
            frequency_dict = json.load(f)
        frequency_matcher(voc1, voc2, tokenizer, frequency_dict)
        print(f"{args.model} - {args.match_type} finished")
        print()

# def spm_match():
#     global args
#     args = parse_args()
#     args.vocab1='/data/sub3/Doo/TokenizationRevisiting/models/originalmodels/plbart/sentencepiece.bpe.model'
#     args.vocab2='/data/sub3/Doo/TokenizationRevisiting/tokenizers/bpe50k/standard.model'
#     args.out_path='/data/sub3/Doo/TokenizationRevisiting/VocabularyTransfer/matches/plbart'
#     args.match_type='tokenizer_match'
#     args.model='plbart'

#     spm1 = sp.SentencePieceProcessor()
#     spm1.load(args.vocab1)
#     vocab1 = {spm1.IdToPiece(i): i for i in range(spm1.GetPieceSize())}

#     spm2 = sp.SentencePieceProcessor()
#     spm2.load(args.vocab2)
#     vocab2 = {spm2.IdToPiece(i): i for i in range(spm2.GetPieceSize())}

#     voc1 = update_dict(vocab1)
#     voc2 = update_dict(vocab2)
#     match_num = matcher(voc1, voc2, tokenizer=spm1)
#     # print(f"match_num: {match_num}")
#     # print(f"match_rate: {match_num/len(voc2)}")
### 新增部分：为 QwenCoder 添加匹配函数 ###
def qwencoder_match():
    """
    为 QwenCoder 或其他基于 AutoTokenizer 的模型处理词汇表匹配。
    注意：你需要将下面的路径替换为你自己的实际路径。
    """
    # --- 请根据您的文件位置修改以下路径 ---
    # 源词汇表（例如，原始 QwenCoder 的 vocab.json）
    args.vocab1 = "/path/to/your/original_qwencoder/vocab.json"
    # 目标词汇表（例如，您训练的新分词器的 vocab.json）
    args.vocab2 = "/path/to/your/new_tokenizer/vocab.json"
    # 匹配结果的输出路径
    args.out_path = "/path/to/your/output_matches/qwencoder"
    
    # QwenCoder 使用的 Tiktoken 分词器通常没有 'Ġ' 前缀
    # 如果你的新分词器有不同的前缀，请在此处修改
    args.vocab1_prefix = ''
    args.vocab2_prefix = ''
    
    # --- 路径修改结束 ---

    print(f"Starting QwenCoder match: {args.match_type}")
    print(f"Source vocab (vocab1): {args.vocab1}")
    print(f"Target vocab (vocab2): {args.vocab2}")
    print(f"Output path: {args.out_path}")

    with open(args.vocab1, 'r', encoding='utf-8') as f:
        vocab1 = json.load(f)
    with open(args.vocab2, 'r', encoding='utf-8') as f:
        vocab2 = json.load(f)

    voc1 = update_dict(vocab1)
    voc2 = update_dict(vocab2)

    if args.match_type == 'tokenizer_match':
        # 从 Hugging Face Hub 或本地路径加载 QwenCoder 的分词器
        # 对于 Qwen 系列模型，通常需要 trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-Coder', trust_remote_code=True)
        tokenizer_matcher(voc1, voc2, tokenizer=tokenizer)
        print(f"{args.model} - {args.match_type} finished")
        print()

    elif args.match_type == 'morphology_match':
        morphology_matcher(voc1, voc2)
        print(f"{args.model} - {args.match_type} finished")
        print()

    elif args.match_type == 'frequency_match':
        # 如果使用频率匹配，你需要提供源模型词元的频率文件
        # --- 如果需要，请修改此频率字典的路径 ---
        args.frequency_dict = "/path/to/your/original_qwencoder/token_frequencies.json"
        
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-Coder', trust_remote_code=True)
        print(f"Loading frequency dictionary from: {args.frequency_dict}")
        with open(args.frequency_dict, 'r', encoding='utf-8') as f:
            frequency_dict = json.load(f)
        frequency_matcher(voc1, voc2, tokenizer, frequency_dict)
        print(f"{args.model} - {args.match_type} finished")
        print()
        
if __name__ == '__main__':
    global args
    args = parse_args()
    
    if args.model=='codebert':
        codebert_match()
    elif args.model=='codet5':
        codet5_match()
    # spm_match()
