# -*- coding: utf-8 -*-

import os
import ujson
from ncc.eval.summarization.summarization_metrics import eval_accuracies
from run.translation.bleu import compute_bleu

CUR_DIR = os.path.dirname(__file__)


def evaluation(src_codes, tgt_codes):
    smoothed_bleu = compute_bleu(
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


file = os.path.join(CUR_DIR, "java-python.pred")
refs, hyps = [], []
with open(file, 'r') as reader:
    for line in reader:
        line = ujson.loads(line)
        refs.append(line['references'])
        hyps.append(line['predictions'][0])

bleu4, smoothed_bleu, rouge_l, meteor = evaluation(hyps, refs)
print(f"{file}, Smoothed BLEU: {smoothed_bleu}, RougeL: {rouge_l}, Meteor: {meteor}, BLEU-4: {bleu4}")

file = os.path.join(CUR_DIR, "python-java.pred")
refs, hyps = [], []
with open(file, 'r') as reader:
    for line in reader:
        line = ujson.loads(line)
        refs.append(line['references'])
        hyps.append(line['predictions'][0])

bleu4, smoothed_bleu, rouge_l, meteor = evaluation(hyps, refs)
print(f"{file}, Smoothed BLEU: {smoothed_bleu}, RougeL: {rouge_l}, Meteor: {meteor}, BLEU-4: {bleu4}")
