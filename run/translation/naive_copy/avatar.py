# -*- coding: utf-8 -*-

import os

from ncc_dataset.avatar.translation import (
    RAW_DIR,
)
from ncc.eval.summarization.smoothed_bleu import compute_smoothed_bleu
from ncc.eval.summarization.summarization_metrics import eval_accuracies
from ncc.utils.file_ops import json_io


def load_data(file, src_lang, tgt_lang):
    with open(file, 'r') as reader:
        src_codes, tgt_codes = [], []
        for line in reader:
            code_snippets = json_io.json_loads(line)
            src_codes.append([code.strip() for code in code_snippets[src_lang]])
            tgt_codes.append([code.strip() for code in code_snippets[tgt_lang]])
        return src_codes, tgt_codes


def evaluation(src_codes, tgt_codes):
    smoothed_bleu = compute_smoothed_bleu(
        reference_corpus=[[code.split() for code in codes] for codes in tgt_codes],
        translation_corpus=[code.split() for code in src_codes],
        smooth=True,
    )[0] * 100

    bleu4, rouge_l, meteor = eval_accuracies(
        hypotheses={i: [code] for i, code in enumerate(src_codes)},
        references={i: codes for i, codes in enumerate(tgt_codes)},
        mode='test',
    )
    bleu4, smoothed_bleu, rouge_l, meteor = map(lambda v: round(v, 2), (bleu4, smoothed_bleu, rouge_l, meteor))
    return bleu4, smoothed_bleu, rouge_l, meteor


if __name__ == '__main__':
    src_lang, tgt_lang = 'java', 'python'
    code_file = os.path.join(RAW_DIR, "test.jsonl")
    SRC_CODES, TGT_CODES = load_data(code_file, src_lang, tgt_lang)

    for idx in [1, 3, 5]:
        print(f"AVATAR-top{idx}")
        # src -> tgt
        src_codes = [codes[0] for codes in SRC_CODES]
        tgt_codes = [codes[:idx] for codes in TGT_CODES]
        bleu4, smoothed_bleu, rouge_l, meteor = evaluation(src_codes, tgt_codes)
        print(
            f"{src_lang} -> {tgt_lang}, Smoothed BLEU: {smoothed_bleu}, RougeL: {rouge_l}, Meteor: {meteor}, BLEU-4: {bleu4}")

        # tgt -> src
        src_codes = [codes[:idx] for codes in SRC_CODES]
        tgt_codes = [codes[0] for codes in TGT_CODES]
        bleu4, smoothed_bleu, rouge_l, meteor = evaluation(tgt_codes, src_codes)
        print(
            f"{tgt_lang} -> {src_lang}, Smoothed BLEU: {smoothed_bleu}, RougeL: {rouge_l}, Meteor: {meteor}, BLEU-4: {bleu4}")
