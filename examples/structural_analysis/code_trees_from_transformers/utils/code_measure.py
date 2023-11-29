import math
import torch
import torch.nn.functional as F

from code_score import Score


class Measure(object):
    def __init__(self, n_layers, n_att):
        self.h_measures = ['cos', 'l1', 'l2']
        self.a_measures = ['hellinger', 'jsd']
        self.a_avg_measures = ['avg_hellinger', 'avg_jsd']
        self.measures = self.h_measures + self.a_measures + self.a_avg_measures
        self.max_m_len = max([len(m) for m in self.measures]) + 2
        self.scores = {m: Score(n_layers) for m in self.h_measures}
        for m in self.a_measures:
            self.scores[m] = Score(n_layers * n_att)
        for m in self.a_avg_measures:
            self.scores[m] = Score(n_layers)

    def derive_dists(self, l_hidden, r_hidden, l_att, r_att):
        syn_dists = {}
        for m in self.h_measures:
            syn_dists[m] = getattr(self, m)(l_hidden, r_hidden)
        for m in self.a_measures:
            syn_dists[m] = getattr(self, m)(l_att, r_att)
            syn_dists[m] = syn_dists[m].view(-1, syn_dists[m].size(-1))
        for m in self.a_avg_measures:
            syn_dists[m] = getattr(self, m)(l_att, r_att)

        return syn_dists

    def derive_final_score(self):
        for m in self.scores.keys():
            self.scores[m].derive_final_score()

    @staticmethod
    def cos(l_hidden, r_hidden):
        # (n_layers, seq_len-1, hidden_dim) * 2 -> (n_layers, seq_len-1)
        return (F.cosine_similarity(l_hidden, r_hidden, dim=-1) + 1) / 2

    @staticmethod
    def l1(l_hidden, r_hidden):
        # (n_layers, seq_len-1, hidden_dim) * 2 -> (n_layers, seq_len-1)
        return torch.norm(l_hidden - r_hidden, p=1, dim=-1)

    @staticmethod
    def l2(l_hidden, r_hidden):
        # (n_layers, seq_len-1, hidden_dim) * 2 -> (n_layers, seq_len-1)
        return torch.norm(l_hidden - r_hidden, p=2, dim=-1)

    @staticmethod
    def kl(p, q):
        eps = 1e-30
        p, q = p + eps, q + eps
        p, q = p / p.sum(dim=-1, keepdim=True), q / q.sum(dim=-1, keepdim=True)
        kl = F.kl_div(torch.log(q), p, reduction='none').sum(dim=-1)
        # kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1)
        # To deal with the numerical instability of the KL-div function in PyTorch
        if (kl < 0).sum() > 0:
            kl = kl * (1 - (kl < 0).float())
        assert torch.isinf(kl).sum() == 0
        assert torch.isnan(kl).sum() == 0
        return kl

    @staticmethod
    def jsd(l_att, r_att):
        m = (l_att + r_att) / 2
        l_kl = Measure.kl(l_att, m)
        r_kl = Measure.kl(r_att, m)
        d = torch.sqrt((l_kl + r_kl) / 2)
        assert (d < 0).sum() == 0
        assert torch.isnan(d).sum() == 0
        return d

    @staticmethod
    def hellinger(l_att, r_att):
        d = (((l_att.sqrt() - r_att.sqrt()) ** 2).sum(dim=-1)).sqrt()
        d /= math.sqrt(2)
        return d

    @staticmethod
    def avg_hellinger(l_att, r_att):
        d = Measure.hellinger(l_att, r_att)
        return d.mean(dim=1)

    @staticmethod
    def avg_jsd(l_att, r_att):
        d = Measure.jsd(l_att, r_att)
        return d.mean(dim=1)