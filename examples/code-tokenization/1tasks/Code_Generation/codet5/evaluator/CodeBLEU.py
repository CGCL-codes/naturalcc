#!/usr/bin/env python3

from codebleu import calc_codebleu


def compute_codebleu(references, predictions, lang="java"):
    return calc_codebleu(
        references,
        predictions,
        lang=lang,
        weights=(0.25, 0.25, 0.25, 0.25),
        tokenizer=None,
    )
