import numpy as np
import torch

from code_yk import get_stats

class Score(object):
    def __init__(self, n):
        self.corpus_f1 = torch.zeros(n, 3, dtype=torch.float)
        self.sent_f1 = torch.zeros(n, dtype=torch.float)
        self.n = n
        self.cnt = 0
        self.labels=['parameters','attribute','argument','list','call','assignment','statement','operator','subscript','block',
                     'clause','parameter']
        self.label_recalls=np.zeros((n,12),dtype=float)
        self.label_cnts=np.zeros(12,dtype=float)

    def update(self, pred_spans, gold_spans, gold_tags):
        pred_sets = [set(ps[:-1]) for ps in pred_spans]
        gold_set = set(gold_spans[:-1])
        self.update_corpus_f1(pred_sets, gold_set)
        self.update_sentence_f1(pred_sets, gold_set)
        self.update_label_recalls(pred_spans, gold_spans, gold_tags)
        self.cnt += 1

    def update_label_recalls(self, pred, gold, tags):
        for i, tag in enumerate(tags):
            if tag not in self.labels:
                continue
            tag_idx = self.labels.index(tag)
            self.label_cnts[tag_idx] += 1
            for z in range(len(pred)):
                if gold[i] in pred[z]:
                    self.label_recalls[z][tag_idx] += 1

    def update_corpus_f1(self, pred, gold):
        stats = torch.tensor([get_stats(pred[i], gold) for i in range(self.n)],
                             dtype=torch.float)
        self.corpus_f1 += stats

    def update_sentence_f1(self, pred, gold):
        # sent-level F1 is based on L83-89 from
        # https://github.com/yikangshen/PRPN/test_phrase_grammar.py
        for i in range(self.n):
            model_out, std_out = pred[i], gold
            overlap = model_out.intersection(std_out)
            prec = float(len(overlap)) / (len(model_out) + 1e-8)
            reca = float(len(overlap)) / (len(std_out) + 1e-8)
            if len(std_out) == 0:
                reca = 1.
                if len(model_out) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            self.sent_f1[i] += f1

    def derive_final_score(self):
        tp = self.corpus_f1[:, 0]
        fp = self.corpus_f1[:, 1]
        fn = self.corpus_f1[:, 2]
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)

        epsilon = 1e-8
        self.corpus_f1 = 2 * prec * recall / (prec + recall + epsilon)
        self.sent_f1 /= self.cnt

        for i in range(len(self.label_recalls)):
            for j in range(len(self.label_recalls[0])):
                self.label_recalls[i][j] /= self.label_cnts[j]