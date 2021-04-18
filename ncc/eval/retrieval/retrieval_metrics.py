import math
import torch


def accuracy(similarity, topk=1, max_order=True):
    if not max_order:
        similarity = -similarity
    _, topk_ids = similarity.topk(topk, dim=-1)
    gt = similarity.new(similarity.size(0)).copy_(
        torch.arange(0, similarity.size(0))
    ).unsqueeze(dim=-1)  # [[0], [1], ..., [B-1]]
    return (gt == topk_ids).sum(-1) / topk


def map(similarity, topk=1, max_order=True):
    """
    MAP@k = mean(1/r_{i}), where i = {1, ..., k}.
    in 1-to-1 retrieval task, only 1 candidate is related.
    """
    if not max_order:
        similarity = -similarity
    _, topk_ids = similarity.topk(topk, dim=-1)
    gt = similarity.new(similarity.size(0)).copy_(
        torch.arange(0, similarity.size(0))
    ).unsqueeze(dim=-1)  # [[0], [1], ..., [B-1]]
    rank = similarity.new(similarity.size(0), topk).copy_(
        (1. / torch.range(1, topk)).expand(similarity.size(0), topk)
    )
    map_k = rank.masked_fill(topk_ids != gt, 0.).mean(dim=-1)
    return map_k


def mrr(similarity, max_order=True):
    if not max_order:
        similarity = -similarity
    gt = similarity.diag()
    ids = similarity >= gt.unsqueeze(dim=-1)
    mrr = 1. / ids.sum(dim=-1)
    return mrr


def ndcg(similarity, topk=1, max_order=True):
    """
        NDCG@k = DCG / IDCG
    where
        DCG = sum_{i}^{k} (2^r_{i} - 1) / log_{2}(i + 1)
        IDCG = sum_{i}^{k} (2^sorted_r_{i} -1) * log2(i + 1)
        sorted_r = sort(r, descending)

    In code retrieval task, relativity between a and b is 1 or 0,
     and only ONE relativity is 1 and others are 0.
    Therefore, r_{i} = {0, 1} and sum(r) = 1.
               sorted_r_{0} = 1 and sorted_r_{j} = 0 where j > 0,
               IDCG = 1,

    """
    if not max_order:
        similarity = -similarity
    _, topk_ids = similarity.topk(topk, dim=-1)
    gt = similarity.new(similarity.size(0)).int().copy_(
        torch.arange(0, similarity.size(0))
    ).unsqueeze(dim=-1)  # [[0], [1], ..., [B-1]]
    rank = similarity.new(similarity.size(0), topk).copy_(
        (torch.range(1, topk)).expand(similarity.size(0), topk)
    )
    rank_mask = topk_ids == gt
    rank = rank.masked_fill(~rank_mask, 0.)
    ndcg = rank_mask.float() * math.log(2) / (rank + 1.).log()
    ndcg = ndcg.masked_fill(~rank_mask, 0.).sum(dim=-1)
    return ndcg
