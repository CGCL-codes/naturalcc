import json
from collections import OrderedDict

from ncc import LOGGER

try:
    from third_party.pycocoevalcap.bleu import corpus_bleu
    from third_party.pycocoevalcap.rouge import Rouge
    from third_party.pycocoevalcap.meteor import Meteor
except ImportError as err:
    LOGGER.warning(err)
    from third_party.download import download

    # download('pycocoevalcap')

from .smoothed_bleu import compute_smoothed_bleu


def eval_accuracies(hypotheses, references, sources=None, filename=None, mode='dev', smoothed_blue=False):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    # Compute BLEU-4 scores
    _, bleu4, ind_bleu = corpus_bleu(hypotheses, references)

    # smoothed blue
    if smoothed_blue:
        refs, hyps = [], []
        for idx in range(len(references)):
            refs.append([line.split() for line in references[idx]])
            hyps.append([line.split() for line in hypotheses[idx]][0])
        sbleu = compute_smoothed_bleu(reference_corpus=refs, translation_corpus=hyps)
    else:
        sbleu = 0.0

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    if mode == 'test':
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    else:
        meteor = 0

    fw = open(filename, 'w') if filename else None
    for key in references.keys():
        if fw:
            pred_i = hypotheses[key]

            logobj = OrderedDict()
            logobj['id'] = key
            if sources is not None:
                logobj['code'] = sources[key]
            logobj['predictions'] = pred_i
            logobj['references'] = references[key]
            logobj['bleu'] = ind_bleu[key]
            logobj['rouge_l'] = ind_rouge[key]
            print(json.dumps(logobj), file=fw)

    if fw: fw.close()

    sbleu, bleu4, rouge_l, meteor = map(lambda score: round(score * 100, 2), (sbleu, bleu4, rouge_l, meteor))
    if smoothed_blue:
        return bleu4, rouge_l, meteor, sbleu
    else:
        return bleu4, rouge_l, meteor
