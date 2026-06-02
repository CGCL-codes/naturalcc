#!/usr/bin/env python3

"""
Lightweight BLEU computation helper adapted from the classic MOSES-style
reference implementation.
"""

import math
import re
import sys
import xml.sax.saxutils

NONORM = 0
PRESERVE_CASE = False
EFF_REF_LEN = "shortest"

NORMALIZE1 = [
    ("<skipped>", ""),
    (r"-\n", ""),
    (r"\n", " "),
]
NORMALIZE1 = [(re.compile(pattern), replace) for (pattern, replace) in NORMALIZE1]

NORMALIZE2 = [
    (r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 "),
    (r"([^0-9])([\.,])", r"\1 \2 "),
    (r"([\.,])([^0-9])", r" \1 \2"),
    (r"([0-9])(-)", r"\1 \2 "),
]
NORMALIZE2 = [(re.compile(pattern), replace) for (pattern, replace) in NORMALIZE2]


def normalize(text):
    if NONORM:
        return text.split()
    if not isinstance(text, str):
        text = " ".join(text)
    for pattern, replace in NORMALIZE1:
        text = re.sub(pattern, replace, text)
    text = xml.sax.saxutils.unescape(text, {"&quot;": '"'})
    text = f" {text} "
    if not PRESERVE_CASE:
        text = text.lower()
    for pattern, replace in NORMALIZE2:
        text = re.sub(pattern, replace, text)
    return text.split()


def count_ngrams(words, n=4):
    counts = {}
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def cook_refs(refs, n=4):
    refs = [normalize(ref) for ref in refs]
    maxcounts = {}
    for ref in refs:
        counts = count_ngrams(ref, n)
        for ngram, count in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
    return [len(ref) for ref in refs], maxcounts


def cook_test(test, item, n=4):
    reflens, refmaxcounts = item
    test = normalize(test)
    result = {"testlen": len(test)}

    if EFF_REF_LEN == "shortest":
        result["reflen"] = min(reflens)
    elif EFF_REF_LEN == "average":
        result["reflen"] = float(sum(reflens)) / len(reflens)
    else:
        result["reflen"] = min(reflens, key=lambda x: abs(x - len(test)))

    result["guess"] = [max(len(test) - k + 1, 0) for k in range(1, n + 1)]
    result["correct"] = [0] * n
    counts = count_ngrams(test, n)
    for ngram, count in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)
    return result


def score_cooked(allcomps, n=4, smooth=1):
    total = {"testlen": 0, "reflen": 0, "guess": [0] * n, "correct": [0] * n}
    for comps in allcomps:
        total["testlen"] += comps["testlen"]
        total["reflen"] += comps["reflen"]
        for key in ("guess", "correct"):
            for k in range(n):
                total[key][k] += comps[key][k]

    logbleu = 0.0
    all_bleus = []
    for k in range(n):
        correct = total["correct"][k]
        guess = total["guess"][k]
        addsmooth = 1 if smooth == 1 and k > 0 else 0
        logbleu += math.log(correct + addsmooth + sys.float_info.min) - math.log(
            guess + addsmooth + sys.float_info.min
        )
        all_bleus.append(-10000000 if guess == 0 else math.log(correct + sys.float_info.min) - math.log(guess))

    logbleu /= float(n)
    all_bleus.insert(0, logbleu)
    brevity_penalty = min(0, 1 - float(total["reflen"] + 1) / (total["testlen"] + 1))
    for i in range(len(all_bleus)):
        if i == 0:
            all_bleus[i] += brevity_penalty
        all_bleus[i] = math.exp(all_bleus[i])
    return all_bleus


def bleu(refs, candidate, smooth=1):
    refs = cook_refs(refs)
    test = cook_test(candidate, refs)
    return score_cooked([test], smooth=smooth)
