#!/usr/bin/env python3

from codebleu import calc_codebleu as _calc_codebleu


class _CalcCodeBleu:
    @staticmethod
    def get_codebleu(gold_fn, output_fn, lang):
        references = [x.strip() for x in open(gold_fn, "r", encoding="utf-8").readlines()]
        predictions = [x.strip() for x in open(output_fn, "r", encoding="utf-8").readlines()]
        return _calc_codebleu(
            references,
            predictions,
            lang=lang,
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None,
        )["codebleu"]


calc_code_bleu = _CalcCodeBleu()
calc_codebleu = calc_code_bleu
