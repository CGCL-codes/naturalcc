from third_party.pycocoevalcap.bleu import corpus_bleu
from third_party.pycocoevalcap.rouge import Rouge
from third_party.pycocoevalcap.meteor import Meteor
from collections import OrderedDict, Counter
import ujson


def eval_accuracies(hypotheses, references, sources=None, filename=None, mode='dev'):
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

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, ind_bleu = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    if mode == 'test':
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    else:
        meteor = 0

    # f1 = AverageMeter()
    # precision = AverageMeter()
    # recall = AverageMeter()

    fw = open(filename, 'w') if filename else None
    for key in references.keys():
        # _prec, _rec, _f1 = compute_eval_score(hypotheses[key][0],
        #                                       references[key])
        # precision.update(_prec)
        # recall.update(_rec)
        # f1.update(_f1)
        if fw:
            # if copy_info is not None and print_copy_info:
            #     prediction = hypotheses[key][0].split()
            #     pred_i = [word + ' [' + str(copy_info[key][j]) + ']'
            #               for j, word in enumerate(prediction)]
            #     pred_i = [' '.join(pred_i)]
            # else:
            pred_i = hypotheses[key]

            logobj = OrderedDict()
            logobj['id'] = key
            if sources is not None:
                logobj['code'] = sources[key]
            logobj['predictions'] = pred_i
            # logobj['references'] = references[key][0] if args.print_one_target \
            #     else references[key]
            logobj['references'] = references[key]
            logobj['bleu'] = ind_bleu[key]
            logobj['rouge_l'] = ind_rouge[key]
            print(json.dumps(logobj), file=fw)

    if fw: fw.close()
    return bleu * 100, rouge_l * 100, meteor * 100  # , precision.avg * 100, recall.avg * 100, f1.avg * 100
