# -*- coding: utf-8 -*-

class RougeScorer(object):
    """
    compute rouge score of string
    >>> rouge_scorer = RougeScorer()
    >>> rouge_scorer.add_string(ref='The dog bit the man.', hyp='The dog bit the man.')
    >>> score = rouge_scorer.score()
    >>> score
    {'rouge-1': {'f': 1.0, 'p': 1.0, 'r': 1.0}, 'rouge-2': {'f': 1.0, 'p': 1.0, 'r': 1.0}, 'rouge-l': {'f': 1.0, 'p': 1.0, 'r': 1.0}}
    """

    def __init__(self, precision=2):
        from rouge import Rouge
        self.rouge = Rouge()
        self._precision = precision
        self.reset()

    def reset(self):
        self.refs = []
        self.hyps = []

    def add_string(self, ref, hyp):
        self.refs.append(ref)
        self.hyps.append(hyp)

    def add_strings(self, refs, hyps):
        self.refs.extend(refs)
        self.hyps.extend(hyps)

    def score(self, avg=True):
        assert len(self.hyps) == len(self.refs) and len(self.refs) > 0
        performance = self.rouge.get_scores(hyps=self.hyps, refs=self.refs, avg=avg)
        return {
            name: {avg_name: round(avg_value, self._precision) for avg_name, avg_value in value.items()}
            for name, value in performance.items()
        }
