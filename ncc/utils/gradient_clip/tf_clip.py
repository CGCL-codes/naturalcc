"""
    TensorFlow functions
"""

import torch


def clip_grad_value_(params, min_value, max_value):
    """
    tensorflow: tf.clip_by_value
    clip gradients among [min_value, max_value]
        Examples:
            torch.Tensor([[1, 1, 2, 4], [3, 4, 8, 5]])
            => tensor([[2., 2., 2., 4.],
                      [3., 4., 5., 5.]])
    """
    assert min_value <= max_value
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [p.grad.detach() for p in filter(lambda p: p.grad is not None, params)]
    if len(grads) == 0:
        if len(params) > 0:
            return params[0].new_tensor(0.)
        else:
            return torch.tensor(0.)

    for g in grads:
        g.data.clamp_(min=min_value, max=max_value)

    return None


def clip_by_norm_(params, clip_norm):
    """
    tensorflow: tf.clip_by_norm
    clip gradients whose l2norm > clip_norm:
        g = g * clip_norm / l2norm(g)

        Examples:
            torch.Tensor([2., 5.]), clip_norm=5
            => tensor([1.8570, 4.6424])
    """
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [p.grad.detach() for p in filter(lambda p: p.grad is not None, params)]
    if len(grads) == 0:
        if len(params) > 0:
            return params[0].new_tensor(0.)
        else:
            return torch.tensor(0.)

    for g in grads:
        g_l2norm = torch.norm(g, p=2, dtype=torch.float).item()
        if g_l2norm > clip_norm:
            g.data.mul_(clip_norm / g_l2norm)
        else:
            continue

    return clip_norm


def clip_by_average_norm_(params, clip_norm):
    """
    tensorflow: tf.clip_by_average_norm
    clip gradients whose l2norm > clip_norm:
        g = g * clip_norm / (l2norm(g) / ||g||)

        Examples:
            torch.Tensor([3., 4.]), clip_norm=5
            => tensor([1.2000, 1.6000]
    """
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [p.grad.detach() for p in filter(lambda p: p.grad is not None, params)]
    if len(grads) == 0:
        if len(params) > 0:
            return params[0].new_tensor(0.)
        else:
            return torch.tensor(0.)

    for g in grads:
        g_avg_l2norm = torch.norm(g, p=2, dtype=torch.float).item() / g.numel()
        g.data.mul_(clip_norm / g_avg_l2norm)


def clip_by_global_norm_(params, clip_norm):
    """
    tensorflow: tf.clip_by_global_norm
    clip gradients if global_norm > clip_norm:
        g = g * clip_norm / max(global_norm, clip_norm)
    where
        global_norm = sqrt(sum(l2norm(g) for g in gradients))

        Examples:
            [torch.Tensor([2., 5.]), torch.Tensor([3., 10.])], clip_norm=5
            => [tensor([0.8513, 2.1281]), [1.2769, 4.2563]]
    """
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [p.grad.detach() for p in filter(lambda p: p.grad is not None, params)]
    if len(grads) == 0:
        if len(params) > 0:
            return params[0].new_tensor(0.)
        else:
            return torch.tensor(0.)

    global_norm = torch.sqrt(torch.sum(
        torch.stack([torch.norm(g, p=2) ** 2 for g in grads]),
        dtype=torch.float,
    ))
    norm_base = float(max(global_norm, clip_norm))

    if global_norm > clip_norm:
        for g in grads:
            g.data.mul_(clip_norm / norm_base)

    return global_norm
