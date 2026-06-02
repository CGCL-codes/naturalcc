#!/usr/bin/env python3

from ...codebert.bleu import bleu


def smooth_bleu(references, prediction):
    return bleu(references, prediction, smooth=1)[0]
