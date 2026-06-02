#!/usr/bin/env python3

import json


def get_tochange_tokens(tokens, affix, length=4):
    to_change_tokens = set()
    for token in tokens:
        if len(token) > length and not token.startswith(affix):
            if affix + token in tokens:
                to_change_tokens.add(token)
    return to_change_tokens


def get_exchange_mapping(vocab, to_change_tokens, affix):
    mapping = {}
    for token in vocab.keys():
        if token in to_change_tokens:
            mapping[vocab[token]] = vocab[affix + token]
            mapping[vocab[affix + token]] = vocab[token]
    return mapping


def get_overuse_mapping(vocab, to_change_tokens, affix, allaffix=True, noaffix=False):
    mapping = {}
    if allaffix:
        for token in vocab.keys():
            if token in to_change_tokens:
                mapping[vocab[token]] = vocab[affix + token]
    if noaffix:
        for token in vocab.keys():
            if token in to_change_tokens:
                mapping[vocab[affix + token]] = vocab[token]
    return mapping


def modify_mapping(ids, mapping):
    for i in range(len(ids)):
        if str(ids[i]) in mapping:
            ids[i] = mapping[str(ids[i])]
    return ids


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(obj, fout, ensure_ascii=False, indent=2)
