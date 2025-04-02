import torch

def Hash1(key):
    key = (~key) + (key << 21)
    key = key ^ (key >> 24)
    key = (key + (key << 3)) + (key << 8)
    key = key ^ (key >> 14)
    key = (key + (key << 2)) + (key << 4)
    key = key ^ (key >> 28)
    key = key + (key << 31)
    return key


def random_shuffle(key, n, device) -> torch.Tensor:
    idxs = torch.arange(n, device=device)
    if not (isinstance(key, int)) and len(key) > 1:
        key = key.unsqueeze(1)
        idxs = idxs.unsqueeze(0)
    values = Hash1(key * n + idxs)
    idxs = torch.sort(values, stable=True, dim=-1)[1]
    return idxs
