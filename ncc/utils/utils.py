# -*- coding: utf-8 -*-

try:
    from amp_C import multi_tensor_l2norm

    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False

import contextlib
import copy
import importlib
import os
import sys
import time
import warnings
from itertools import accumulate
from typing import (
    List, Dict, Optional,
)

import torch
import torch.nn.functional as F
from torch import Tensor

from ncc import LOGGER
from ncc.data.constants import (
    PAD, EOS, UNK,
)
from ncc.utils.logging.meters import safe_round
from dgl import DGLGraph


def now() -> str:
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, DGLGraph):
            return x.to(torch.device('cuda'))
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def clean_up_sentence(sentence: torch.Tensor, remove_UNK=False, remove_EOS=False):
    # cut PAD at the sentence's end
    sentence = sentence.tolist()
    if PAD in sentence:
        sentence = sentence[:sentence.index(PAD)]
    # cut EOS at the sentence's end
    if EOS in sentence:
        sentence = sentence[:sentence.index(EOS) + 1]
    # filter UNK
    if remove_UNK:
        sentence = filter(lambda x: x != UNK, sentence)
    # remove EOS
    if remove_EOS:
        if len(sentence) > 0 and sentence[-1] == EOS:
            sentence = sentence[:-1]

    return sentence


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def split_paths(paths: str) -> List[str]:
    return paths.split(os.pathsep) if "://" not in paths else paths.split("|")


def load_ensemble_for_inference(filenames, task, model_arg_overrides=None):
    from ncc.utils import checkpoint_utils

    deprecation_warning(
        "utils.load_ensemble_for_inference is deprecated. "
        "Please use checkpoint_utils.load_model_ensemble instead."
    )
    return checkpoint_utils.load_model_ensemble(
        filenames, arg_overrides=model_arg_overrides, task=task
    )


def get_incremental_state(
    module,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    key: str,
) -> Optional[Dict[str, Optional[Tensor]]]:
    """Helper for getting incremental state for an nn.Module."""
    return module.get_incremental_state(incremental_state, key)


def set_incremental_state(
    module,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    key: str,
    value: Dict[str, Optional[Tensor]],
) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        result = module.set_incremental_state(incremental_state, key, value)
        if result is not None:
            incremental_state = result
    return incremental_state


def load_align_dict(replace_unk):
    if replace_unk is None:
        align_dict = None
    elif isinstance(replace_unk, str) and len(replace_unk) > 0:
        # Load alignment dictionary for unknown word replacement if it was passed as an argument.
        align_dict = {}
        with open(replace_unk, "r") as f:
            for line in f:
                cols = line.split()
                align_dict[cols[0]] = cols[1]
    else:
        # No alignment dictionary provided but we still want to perform unknown word replacement by copying the
        # original source word.
        align_dict = {}
    return align_dict


def print_embed_overlap(embed_dict, vocab_dict):
    embed_keys = set(embed_dict.keys())
    vocab_keys = set(vocab_dict.symbols)
    overlap = len(embed_keys & vocab_keys)
    LOGGER.info("found {}/{} types in embedding file".format(overlap, len(vocab_dict)))


def make_positions(tensor, padding_idx: int):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    if padding_idx is None:
        positional_ids = tensor.new(*tensor.size())
        positional_ids.data.copy_(torch.arange(start=0, end=tensor.size(1)).expand(tensor.size()))
        return positional_ids
    else:
        mask = tensor.ne(padding_idx).int()
        # mask = torch.ones(tensor.size()).int().cuda()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def make_positions_hibert(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def strip_eos(tensor, eos):
    assert tensor.dim() == 1
    if eos in tensor:
        idx = tensor.tolist().index(eos)
        return tensor[:idx]
    else:
        return tensor


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def convert_padding_direction(
    src_tokens, padding_idx, right_to_left=False, left_to_right=False
):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


def item(tensor):
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


def multi_tensor_total_norm(grads, chunk_size=2048 * 32) -> torch.Tensor:
    per_device_grads = {}
    norms = []
    for grad in grads:
        device = grad.device
        cur_device_grads = per_device_grads.get(device)
        if cur_device_grads is None:
            cur_device_grads = []
            per_device_grads[device] = cur_device_grads
        cur_device_grads.append(grad)
    for device in per_device_grads.keys():
        cur_device_grads = per_device_grads[device]
        if device.type == "cuda":
            # TODO(msb) return has_inf
            has_inf = torch.zeros((1, 1), dtype=torch.int, device=device)
            with torch.cuda.device(device):
                norm = multi_tensor_l2norm(chunk_size, has_inf, [cur_device_grads], False)
                norms.append(norm[0])
        else:
            norms += [torch.norm(g, p=2, dtype=torch.float32) for g in cur_device_grads]
    total_norm = torch.norm(torch.stack(norms))
    return total_norm


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _match_types(arg1, arg2):
    """Convert the numerical argument to the same type as the other argument"""

    def upgrade(arg_number, arg_structure):
        if isinstance(arg_structure, tuple):
            return (arg_number, arg_number)
        elif isinstance(arg_structure, dict):
            arg = copy.deepcopy(arg_structure)
            for k in arg:
                arg[k] = upgrade(arg_number, arg_structure[k])
            return arg
        else:
            return arg_number

    if isinstance(arg1, float) or isinstance(arg1, int):
        return upgrade(arg1, arg2), arg2
    elif isinstance(arg2, float) or isinstance(arg2, int):
        return arg1, upgrade(arg2, arg1)

    return arg1, arg2


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            max_positions, arg = _match_types(max_positions, arg)
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(map(nullsafe_min, zip(max_positions, arg)))

    return max_positions


def import_user_module(args):
    module_path = getattr(args, "user_dir", None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        if not os.path.exists(module_path):
            fairseq_rel_path = os.path.join(
                os.path.dirname(__file__), "..", args.user_dir
            )
            if os.path.exists(fairseq_rel_path):
                module_path = fairseq_rel_path
        module_parent, module_name = os.path.split(module_path)

        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            importlib.import_module(module_name)
            sys.path.pop(0)


def get_perplexity(loss, round=2, base=2):
    if loss is None:
        return 0.
    try:
        return safe_round(base ** loss, round)
    except OverflowError:
        return float('inf')


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


@contextlib.contextmanager
def eval(model):
    is_training = model.training
    model.eval()
    yield
    model.train(is_training)


def has_parameters(module):
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


def parse_alignment(line):
    """
    Parses a single line from the alingment file.

    Args:
        line (str): String containing the alignment of the format:
            <src_idx_1>-<tgt_idx_1> <src_idx_2>-<tgt_idx_2> ..
            <src_idx_m>-<tgt_idx_m>. All indices are 0 indexed.

    Returns:
        torch.IntTensor: packed alignments of shape (2 * m).
    """
    alignments = line.strip().split()
    parsed_alignment = torch.IntTensor(2 * len(alignments))
    for idx, alignment in enumerate(alignments):
        src_idx, tgt_idx = alignment.split("-")
        parsed_alignment[2 * idx] = int(src_idx)
        parsed_alignment[2 * idx + 1] = int(tgt_idx)
    return parsed_alignment


def get_token_to_word_mapping(tokens, exclude_list):
    n = len(tokens)
    word_start = [int(token not in exclude_list) for token in tokens]
    word_idx = list(accumulate(word_start))
    token_to_word = {i: word_idx[i] for i in range(n)}
    return token_to_word


def extract_hard_alignment(attn, src_sent, tgt_sent, pad, eos):
    tgt_valid = ((tgt_sent != pad) & (tgt_sent != eos)).nonzero().squeeze(dim=-1)
    src_invalid = ((src_sent == pad) | (src_sent == eos)).nonzero().squeeze(dim=-1)
    src_token_to_word = get_token_to_word_mapping(src_sent, [eos, pad])
    tgt_token_to_word = get_token_to_word_mapping(tgt_sent, [eos, pad])
    alignment = []
    if len(tgt_valid) != 0 and len(src_invalid) < len(src_sent):
        attn_valid = attn[tgt_valid]
        attn_valid[:, src_invalid] = float("-inf")
        _, src_indices = attn_valid.max(dim=1)
        for tgt_idx, src_idx in zip(tgt_valid, src_indices):
            alignment.append(
                (
                    src_token_to_word[src_idx.item()] - 1,
                    tgt_token_to_word[tgt_idx.item()] - 1,
                )
            )
    return alignment


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]]
            )
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding
