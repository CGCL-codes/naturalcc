import json
import sentencepiece as sp
def get_tochange_tokens(tokens, affix, length=4):
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
    mapping = {}
    for token in vocab.keys():
        if token in to_change_tokens:
            mapping[vocab[token]]=vocab[affix+token]
            mapping[vocab[affix+token]]=vocab[token]
    return mapping
def get_overuse_mapping(vocab, to_change_tokens, affix, allaffix=True, noaffix=False):
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
    for i in range(len(ids)):
        if ids[i] in mapping.keys():
            ids[i]=mapping[ids[i]]
    return ids


if __name__ == '__main__':

    original_vocab_path = './vocab.json'
    with open(original_vocab_path, 'r') as f1, open('./mapping/codet5/exchange.json', 'w', encoding='utf-8') as f2, open('./mapping/codet5/overuse-allaffix.json', 'w', encoding='utf-8') as f3, open('./mapping/codet5/overuse-noaffix.json', 'w', encoding='utf-8') as f4:
        vocab = json.load(f1)
        to_change_tokens = get_tochange_tokens(vocab, affix='Ġ', length=4)

        exchange_mapping = get_exchange_mapping(vocab, to_change_tokens, affix='Ġ')
        json.dump(exchange_mapping, f2)

        overuse_allaffix_mapping = get_overuse_mapping(vocab, to_change_tokens, affix='Ġ', allaffix=True, noaffix=False)
        json.dump(overuse_allaffix_mapping, f3)

        overuse_noaffix_mapping = get_overuse_mapping(vocab, to_change_tokens, affix='Ġ', allaffix=False, noaffix=True)
        json.dump(overuse_noaffix_mapping, f4)
