# -*- coding: utf-8 -*-

import os

from preprocess.avatar.translation import (
    ATTRIBUTES_DIR,
)
from ncc.eval.summarization.smoothed_bleu import compute_smoothed_bleu
from ncc.eval.summarization.summarization_metrics import eval_accuracies
from ncc.utils.file_ops import json_io


def load_data(file):
    with open(file, 'r') as reader:
        codes = [
            json_io.json_loads(line)
            for line in reader
        ]
        return codes


def evaluation(src_codes, tgt_codes):
    smoothed_bleu = compute_smoothed_bleu(
        reference_corpus=[[code.split()] for code in tgt_codes],
        translation_corpus=[code.split() for code in src_codes],
        smooth=True,
    )[0] * 100

    bleu4, rouge_l, meteor = eval_accuracies(
        hypotheses={i: [code] for i, code in enumerate(src_codes)},
        references={i: [code] for i, code in enumerate(tgt_codes)},
        mode='test',
    )
    bleu4, smoothed_bleu, rouge_l, meteor = map(lambda v: round(v, 2), (bleu4, smoothed_bleu, rouge_l, meteor))
    return bleu4, smoothed_bleu, rouge_l, meteor


if __name__ == '__main__':
    src_lang, tgt_lang = 'java', 'csharp'
    src_file = os.path.join(ATTRIBUTES_DIR, src_lang, f"test.code")
    tgt_file = os.path.join(ATTRIBUTES_DIR, tgt_lang, f"test.code")
    src_codes = load_data(src_file)
    tgt_codes = load_data(tgt_file)

    bleu4, smoothed_bleu, rouge_l, meteor = evaluation(src_codes, tgt_codes)
    print(
        f"{src_lang} -> {tgt_lang}, Smoothed BLEU: {smoothed_bleu}, RougeL: {rouge_l}, Meteor: {meteor}, BLEU-4: {bleu4}")

    bleu4, smoothed_bleu, rouge_l, meteor = evaluation(tgt_codes, src_codes)
    print(
        f"{tgt_lang} -> {src_lang}, Smoothed BLEU: {smoothed_bleu}, RougeL: {rouge_l}, Meteor: {meteor}, BLEU-4: {bleu4}")
