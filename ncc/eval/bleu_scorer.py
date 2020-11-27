# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import math
import torch

try:
    from ncc import libbleu
except ImportError as e:
    import sys

    sys.stderr.write('ERROR: missing libbleu.so. run `pip install --editable .`\n')
    raise e

C = ctypes.cdll.LoadLibrary(libbleu.__file__)


class BleuStat(ctypes.Structure):
    _fields_ = [
        ('reflen', ctypes.c_size_t),
        ('predlen', ctypes.c_size_t),
        ('match1', ctypes.c_size_t),
        ('count1', ctypes.c_size_t),
        ('match2', ctypes.c_size_t),
        ('count2', ctypes.c_size_t),
        ('match3', ctypes.c_size_t),
        ('count3', ctypes.c_size_t),
        ('match4', ctypes.c_size_t),
        ('count4', ctypes.c_size_t),
    ]


class SacrebleuScorer(object):
    """
    compute bleu score of string.
    Examples:
    >>> bleu_metric = SacrebleuScorer()
    >>> bleu_metric.add_string(ref='The dog bit the man.', pred='The dog bit the man.') # 100.00000000000004
    >>> bleu_metric.add_string(ref='The dog had bit the man.', pred='The dog bit the man.') # 51.15078115793242
    >>> bleu_score = bleu_metric.score() # avg: 75.35497352995401
    >>> bleu_score
    75.35497352995401

    References:
    >>> import sacrebleu
    >>> refs = [['The dog bit the man.'], ['The dog had bit the man.'],]
    >>> sys = ['The dog bit the man.']
    >>> bleu = sacrebleu.corpus_bleu(sys, refs)
    >>> bleu.score # 100.00000000000004
    100.00000000000004
    """

    def __init__(self):
        import sacrebleu
        self.sacrebleu = sacrebleu
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            raise NotImplementedError
        self.ref = []
        self.sys = []

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.sys.append(pred)

    def score(self, order=4):
        return self.result_string(order).score

    def result_string(self, order=4):
        if order != 4:
            raise NotImplementedError
        return self.sacrebleu.corpus_bleu(self.sys, [self.ref])


class NLTKBleuScorer(object):
    """
    compute bleu score of string.
    Examples:
    >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...               'ensures', 'that', 'the', 'military', 'always',
    ...               'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...              'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...              'heed', 'Party', 'commands']
    >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...              'guarantees', 'the', 'military', 'forces', 'always',
    ...              'being', 'under', 'the', 'command', 'of', 'the',
    ...              'Party']
    >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...              'army', 'always', 'to', 'heed', 'the', 'directions',
    ...              'of', 'the', 'party']
    >>> bleu_scorer = NLTKBleuScorer()
    >>> bleu_scorer.add_string(
    ...    refs=[reference1, reference2, reference3],
    ...    pred=hypothesis1,
    ... )
    >>> score = bleu_scorer.score()
    >>> score
    {'BLEU-1': 94.44, 'BLEU-2': 74.54, 'BLEU-3': 62.41, 'BLEU-4': 50.46}

    References:
    >>> from nltk.translate.bleu_score import corpus_bleu
    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...        'ensures', 'that', 'the', 'military', 'always',
    ...        'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...         'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...         'heed', 'Party', 'commands']
    >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...         'guarantees', 'the', 'military', 'forces', 'always',
    ...         'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...         'army', 'always', 'to', 'heed', 'the', 'directions',
    ...         'of', 'the', 'party']
    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
    ...        'interested', 'in', 'world', 'history']
    >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
    ...         'because', 'he', 'read', 'the', 'book']
    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    >>> hypotheses = [hyp1, hyp2]
    >>> bleu1 = corpus_bleu(list_of_references, hypotheses, weights=(1, 0, 0, 0))  # doctest: +ELLIPSIS
    >>> bleu2 = corpus_bleu(list_of_references, hypotheses, weights=(0.5, 0.5, 0, 0))  # doctest: +ELLIPSIS
    >>> bleu3 = corpus_bleu(list_of_references, hypotheses, weights=(0.33, 0.33, 0.33, 0))  # doctest: +ELLIPSIS
    >>> bleu4 = corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))  # doctest: +ELLIPSIS
    >>> bleu1, bleu2, bleu3, bleu4
    (0.9655172413793104, 0.8242803277698696, 0.7093989814471865, 0.5920778868801042)
    """

    def __init__(self, N_gram=4, precision=2):
        from nltk.translate import bleu_score
        self.bleu_score = bleu_score
        self.N_gram = N_gram
        self._precision = precision
        self._weights = [
            tuple([round(1 / i, 4)] * i + [0.] * (N_gram - i))
            for i in range(1, N_gram + 1)
        ]
        self.smoothing_fn = bleu_score.SmoothingFunction().method3
        self.reset()

    def reset(self):
        self.refs = []
        self.sys = []

    def add_string(self, refs, pred):
        if isinstance(refs, str):
            refs = [refs.split()]
        if isinstance(pred, str):
            pred = pred.split()
        self.refs.append(refs)
        self.sys.append(pred)

    def sentence_score(self, refs, pred):
        performance = {'BLEU-{}'.format(i): None for i in range(1, self.N_gram + 1)}
        for idx, weights in enumerate(self._weights, start=1):
            bleu_score = self.bleu_score.sentence_bleu(refs, pred, weights=weights, \
                                                       smoothing_function=self.smoothing_fn)
            performance['BLEU-{}'.format(idx)] = round(bleu_score * 100, self._precision)
        return performance

    def score(self):
        return self.result_string()

    def result_string(self):
        performance = {'BLEU-{}'.format(i): None for i in range(1, self.N_gram + 1)}
        for idx, weights in enumerate(self._weights, start=1):
            bleu_score = self.bleu_score.corpus_bleu(self.refs, self.sys, weights=weights, \
                                                     smoothing_function=self.smoothing_fn)
            performance['BLEU-{}'.format(idx)] = round(bleu_score * 100, self._precision)
        return performance


class Scorer(object):
    """
    compute bleu score of TorchTensor.
    """

    def __init__(self, pad, eos, unk):
        self.stat = BleuStat()
        self.pad = pad
        self.eos = eos
        self.unk = unk
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            C.bleu_one_init(ctypes.byref(self.stat))
        else:
            C.bleu_zero_init(ctypes.byref(self.stat))

    def add(self, ref, pred):
        if not isinstance(ref, torch.IntTensor):
            raise TypeError('ref must be a torch.IntTensor (got {})'
                            .format(type(ref)))
        if not isinstance(pred, torch.IntTensor):
            raise TypeError('pred must be a torch.IntTensor(got {})'
                            .format(type(pred)))

        # don't match unknown words
        rref = ref.clone()
        assert not rref.lt(0).any()
        rref[rref.eq(self.unk)] = -999

        rref = rref.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(rref.size(0)),
            ctypes.c_void_p(rref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(self.pad),
            ctypes.c_int(self.eos))

    def score(self, order=4):
        psum = sum(math.log(p) if p > 0 else float('-Inf')
                   for p in self.precision()[:order])
        return self.brevity() * math.exp(psum / order) * 100

    def precision(self):
        def ratio(a, b):
            return a / b if b > 0 else 0

        return [
            ratio(self.stat.match1, self.stat.count1),
            ratio(self.stat.match2, self.stat.count2),
            ratio(self.stat.match3, self.stat.count3),
            ratio(self.stat.match4, self.stat.count4),
        ]

    def brevity(self):
        r = self.stat.reflen / self.stat.predlen
        return min(1, math.exp(1 - r))

    def result_string(self, order=4):
        assert order <= 4, "BLEU scores for order > 4 aren't supported"
        fmt = 'BLEU{} = {:2.2f}, {:2.1f}'
        for _ in range(1, order):
            fmt += '/{:2.1f}'
        fmt += ' (BP={:.3f}, ratio={:.3f}, syslen={}, reflen={})'
        bleup = [p * 100 for p in self.precision()[:order]]
        return fmt.format(order, self.score(order=order), *bleup,
                          self.brevity(), self.stat.predlen / self.stat.reflen,
                          self.stat.predlen, self.stat.reflen)
