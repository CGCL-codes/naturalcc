# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from ncc.utils import utils
from ncc.data.indexed_dataset import get_available_dataset_impl
from ncc.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    EvalLMConfig,
    GenerationConfig,
    InteractiveConfig,
    OptimizationConfig,
    EMAConfig,
)
from ncc.dataclass.utils import gen_parser_from_dataclass

# this import is for backward compatibility
from ncc.utils.utils import csv_str_list, eval_bool, eval_str_dict, eval_str_list  # noqa


def get_preprocessing_parser(default_task="translation"):
    parser = get_parser("Preprocessing", default_task)
    add_preprocess_args(parser)
    return parser


def get_training_parser(default_task="translation"):
    parser = get_parser("Trainer", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    add_ema_args(parser)
    return parser


def get_generation_parser(interactive=False, default_task="translation"):
    parser = get_parser("Generation", default_task)
    add_dataset_args(parser, gen=True)
    add_distributed_training_args(parser, default_world_size=1)
    add_generation_args(parser)
    add_checkpoint_args(parser)
    if interactive:
        add_interactive_args(parser)
    return parser


def get_speech_generation_parser(default_task="text_to_speech"):
    parser = get_parser("Speech Generation", default_task)
    add_dataset_args(parser, gen=True)
    add_distributed_training_args(parser, default_world_size=1)
    add_speech_generation_args(parser)
    return parser


def get_interactive_generation_parser(default_task="translation"):
    return get_generation_parser(interactive=True, default_task=default_task)


def get_eval_lm_parser(default_task="language_modeling"):
    parser = get_parser("Evaluate Language Model", default_task)
    add_dataset_args(parser, gen=True)
    add_distributed_training_args(parser, default_world_size=1)
    add_eval_lm_args(parser)
    return parser


def get_validation_parser(default_task=None):
    parser = get_parser("Validation", default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser, default_world_size=1)
    group = parser.add_argument_group("Evaluation")
    gen_parser_from_dataclass(group, CommonEvalConfig())
    return parser


def parse_args_and_arch(
    parser: argparse.ArgumentParser,
    input_args: List[str] = None,
    parse_known: bool = False,
    suppress_defaults: bool = False,
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
):
    """
    Args:
        parser (ArgumentParser): the parser
        input_args (List[str]): strings to parse, defaults to sys.argv
        parse_known (bool): only parse known arguments, similar to
            `ArgumentParser.parse_known_args`
        suppress_defaults (bool): parse while ignoring all default values
        modify_parser (Optional[Callable[[ArgumentParser], None]]):
            function to modify the parser, e.g., to set default values
    """
    if suppress_defaults:
        # Parse args without any default values. This requires us to parse
        # twice, once to identify all the necessary task/model args, and a second
        # time with all defaults set to None.
        args = parse_args_and_arch(
            parser,
            input_args=input_args,
            parse_known=parse_known,
            suppress_defaults=False,
        )
        suppressed_parser = argparse.ArgumentParser(add_help=False, parents=[parser])
        suppressed_parser.set_defaults(**{k: None for k, v in vars(args).items()})
        args = suppressed_parser.parse_args(input_args)
        return argparse.Namespace(
            **{k: v for k, v in vars(args).items() if v is not None}
        )

    from ncc.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY, MODEL_REGISTRY

    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args(input_args)
    utils.import_user_module(usr_args)

    if modify_parser is not None:
        modify_parser(parser)

    # The parser doesn't know about model/criterion/optimizer-specific args, so
    # we parse twice. First we parse the model/criterion/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser.
    if hasattr(args, "arch"):
        model_specific_group = parser.add_argument_group(
            "Model-specific configuration",
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        if args.arch in ARCH_MODEL_REGISTRY:
            ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group)
        elif args.arch in MODEL_REGISTRY:
            MODEL_REGISTRY[args.arch].add_args(model_specific_group)
        else:
            raise RuntimeError()

    if hasattr(args, "task"):
        from ncc.tasks import TASK_REGISTRY

        TASK_REGISTRY[args.task].add_args(parser)
    if getattr(args, "use_bmuf", False):
        # hack to support extra args for block distributed data parallelism
        from ncc.optim.bmuf import NccBMUF

        NccBMUF.add_args(parser)

    # Add *-specific args to parser.
    from ncc.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        choice = getattr(args, registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            if hasattr(cls, "add_args"):
                cls.add_args(parser)
            elif hasattr(cls, "__dataclass"):
                gen_parser_from_dataclass(parser, cls.__dataclass())

    # Modify the parser a second time, since defaults may have been reset
    if modify_parser is not None:
        modify_parser(parser)

    # Parse a second time.
    if parse_known:
        args, extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None
    # Post-process args.
    if (
        hasattr(args, "batch_size_valid") and args.batch_size_valid is None
    ) or not hasattr(args, "batch_size_valid"):
        args.batch_size_valid = args.batch_size
    if hasattr(args, "max_tokens_valid") and args.max_tokens_valid is None:
        args.max_tokens_valid = args.max_tokens
    if getattr(args, "memory_efficient_fp16", False):
        args.fp16 = True
    if getattr(args, "memory_efficient_bf16", False):
        args.bf16 = True
    args.tpu = getattr(args, "tpu", False)
    args.bf16 = getattr(args, "bf16", False)
    if args.bf16:
        args.tpu = True
    if args.tpu and args.fp16:
        raise ValueError("Cannot combine --fp16 and --tpu, use --bf16 on TPUs")

    if getattr(args, "seed", None) is None:
        args.seed = 1  # default seed for training
        args.no_seed_provided = True
    else:
        args.no_seed_provided = False

    if getattr(args, "update_epoch_batch_itr", None) is None:
        if hasattr(args, "grouped_shuffling"):
            args.update_epoch_batch_itr = args.grouped_shuffling
        else:
            args.grouped_shuffling = False
            args.update_epoch_batch_itr = False

    # Apply architecture configuration.
    if hasattr(args, "arch") and args.arch in ARCH_CONFIG_REGISTRY:
        ARCH_CONFIG_REGISTRY[args.arch](args)

    if parse_known:
        return args, extra
    else:
        return args


def get_parser(desc, default_task="translation"):
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)
    usr_args, _ = usr_parser.parse_known_args()
    utils.import_user_module(usr_args)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    gen_parser_from_dataclass(parser, CommonConfig())

    from ncc.registry import REGISTRIES

    for registry_name, REGISTRY in REGISTRIES.items():
        parser.add_argument(
            "--" + registry_name.replace("_", "-"),
            default=REGISTRY["default"],
            choices=REGISTRY["registry"].keys(),
        )

    # Task definitions can be found under fairseq/tasks/
    from ncc.tasks import TASK_REGISTRY

    parser.add_argument(
        "--task",
        metavar="TASK",
        default=default_task,
        choices=TASK_REGISTRY.keys(),
        help="task",
    )
    # fmt: on
    return parser


def add_preprocess_args(parser):
    group = parser.add_argument_group("Preprocessing")
    # fmt: off
    group.add_argument("-s", "--source-lang", default=None, metavar="SRC",
                       help="source language")
    group.add_argument("-t", "--target-lang", default=None, metavar="TARGET",
                       help="target language")
    group.add_argument("--trainpref", metavar="FP", default=None,
                       help="train file prefix (also used to build dictionaries)")
    group.add_argument("--validpref", metavar="FP", default=None,
                       help="comma separated, valid file prefixes "
                            "(words missing from train set are replaced with <unk>)")
    group.add_argument("--testpref", metavar="FP", default=None,
                       help="comma separated, test file prefixes "
                            "(words missing from train set are replaced with <unk>)")
    group.add_argument("--align-suffix", metavar="FP", default=None,
                       help="alignment file suffix")
    group.add_argument("--destdir", metavar="DIR", default="data-bin",
                       help="destination dir")
    group.add_argument("--thresholdtgt", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    group.add_argument("--thresholdsrc", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    group.add_argument("--tgtdict", metavar="FP",
                       help="reuse given target dictionary")
    group.add_argument("--srcdict", metavar="FP",
                       help="reuse given source dictionary")
    group.add_argument("--nwordstgt", metavar="N", default=-1, type=int,
                       help="number of target words to retain")
    group.add_argument("--nwordssrc", metavar="N", default=-1, type=int,
                       help="number of source words to retain")
    group.add_argument("--alignfile", metavar="ALIGN", default=None,
                       help="an alignment file (optional)")
    parser.add_argument('--dataset-impl', metavar='FORMAT', default='mmap',
                        choices=get_available_dataset_impl(),
                        help='output dataset implementation')
    group.add_argument("--joined-dictionary", action="store_true",
                       help="Generate joined dictionary")
    group.add_argument("--only-source", action="store_true",
                       help="Only process the source language")
    group.add_argument("--padding-factor", metavar="N", default=8, type=int,
                       help="Pad dictionary size to be multiple of N")
    group.add_argument("--workers", metavar="N", default=1, type=int,
                       help="number of parallel workers")
    group.add_argument("--dict-only", action='store_true',
                       help="if true, only builds a dictionary and then exits")
    # fmt: on
    return parser


def add_dataset_args(parser, train=False, gen=False):
    group = parser.add_argument_group("dataset_data_loading")
    gen_parser_from_dataclass(group, DatasetConfig())
    # fmt: on
    return group


def add_distributed_training_args(parser, default_world_size=None):
    group = parser.add_argument_group("distributed_training")
    if default_world_size is None:
        default_world_size = max(1, torch.cuda.device_count())
    gen_parser_from_dataclass(
        group, DistributedTrainingConfig(distributed_world_size=default_world_size)
    )
    return group


def add_optimization_args(parser):
    group = parser.add_argument_group("optimization")
    # fmt: off
    gen_parser_from_dataclass(group, OptimizationConfig())
    # fmt: on
    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group("checkpoint")
    # fmt: off
    gen_parser_from_dataclass(group, CheckpointConfig())
    # fmt: on
    return group


def add_common_eval_args(group):
    gen_parser_from_dataclass(group, CommonEvalConfig())


def add_eval_lm_args(parser):
    group = parser.add_argument_group("LM Evaluation")
    add_common_eval_args(group)
    gen_parser_from_dataclass(group, EvalLMConfig())


def add_generation_args(parser):
    group = parser.add_argument_group("Generation")
    add_common_eval_args(group)
    gen_parser_from_dataclass(group, GenerationConfig())
    return group


def add_speech_generation_args(parser):
    group = parser.add_argument_group("Speech Generation")
    add_common_eval_args(group)  # NOTE: remove_bpe is not needed
    # fmt: off
    group.add_argument('--eos_prob_threshold', default=0.5, type=float,
                       help='terminate when eos probability exceeds this')
    # fmt: on
    return group


def add_interactive_args(parser):
    group = parser.add_argument_group("Interactive")
    gen_parser_from_dataclass(group, InteractiveConfig())


def add_model_args(parser):
    group = parser.add_argument_group("Model configuration")
    # fmt: off

    # Model definitions can be found under fairseq/models/
    #
    # The model architecture can be specified in several ways.
    # In increasing order of priority:
    # 1) model defaults (lowest priority)
    # 2) --arch argument
    # 3) --encoder/decoder-* arguments (highest priority)
    from ncc.models import ARCH_MODEL_REGISTRY
    group.add_argument('--arch', '-a', metavar='ARCH',
                       choices=ARCH_MODEL_REGISTRY.keys(),
                       help='model architecture')
    # fmt: on
    return group


def get_args(
    data: Union[str, Path],
    task: str = "translation",
    arch: str = "transformer",
    **overrides
):
    parser = get_training_parser(task)
    args = parse_args_and_arch(parser, [str(data), "--task", task, "--arch", arch])

    for k, v in overrides.items():
        setattr(args, k, v)

    return args


def add_ema_args(parser):
    group = parser.add_argument_group("EMA configuration")
    gen_parser_from_dataclass(group, EMAConfig())


# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# import argparse
# import collections
# import contextlib
# import copy
# import importlib
# import logging
# import os
# import sys
# import warnings
# from itertools import accumulate
# from typing import TYPE_CHECKING, Callable, Dict, List, Optional

# import torch
# import torch.nn.functional as F
# from torch import Tensor

# if TYPE_CHECKING:
#     from ncc.modules.multihead_attention import MultiheadAttention

# try:
#     from amp_C import multi_tensor_l2norm

#     multi_tensor_l2norm_available = True
# except ImportError:
#     multi_tensor_l2norm_available = False

# try:
#     import torch_xla.core.xla_model as xm
# except ImportError:
#     xm = None


# logger = logging.getLogger(__name__)


# MANIFOLD_PATH_SEP = "|"


# class FileContentsAction(argparse.Action):
#     def __init__(self, option_strings, dest, nargs=None, **kwargs):
#         if nargs is not None:
#             raise ValueError("nargs not allowed")
#         super(FileContentsAction, self).__init__(option_strings, dest, **kwargs)

#     def __call__(self, parser, namespace, values, option_string=None):
#         from ncc.file_io import PathManager

#         if PathManager.isfile(values):
#             with PathManager.open(values) as f:
#                 argument = f.read().strip()
#         else:
#             argument = values
#         setattr(namespace, self.dest, argument)


# def split_paths(paths: str, separator=os.pathsep) -> List[str]:
#     return (
#         paths.split(separator) if "://" not in paths else paths.split(MANIFOLD_PATH_SEP)
#     )


# def load_ensemble_for_inference(filenames, task, model_arg_overrides=None):
#     from ncc import checkpoint_utils

#     deprecation_warning(
#         "utils.load_ensemble_for_inference is deprecated. "
#         "Please use checkpoint_utils.load_model_ensemble instead."
#     )
#     return checkpoint_utils.load_model_ensemble(
#         filenames, arg_overrides=model_arg_overrides, task=task
#     )


# def apply_to_sample(f, sample):
#     if hasattr(sample, "__len__") and len(sample) == 0:
#         return {}

#     def _apply(x):
#         if torch.is_tensor(x):
#             return f(x)
#         elif isinstance(x, collections.OrderedDict):
#             # OrderedDict has attributes that needs to be preserved
#             od = collections.OrderedDict(
#                 (key, _apply(value)) for key, value in x.items()
#             )
#             od.__dict__ = x.__dict__
#             return od
#         elif isinstance(x, dict):
#             return {key: _apply(value) for key, value in x.items()}
#         elif isinstance(x, list):
#             return [_apply(x) for x in x]
#         elif isinstance(x, tuple):
#             return tuple(_apply(x) for x in x)
#         elif isinstance(x, set):
#             return {_apply(x) for x in x}
#         else:
#             return x

#     return _apply(sample)


# def move_to_cuda(sample, device=None):
#     device = device or torch.cuda.current_device()

#     def _move_to_cuda(tensor):
#         # non_blocking is ignored if tensor is not pinned, so we can always set
#         # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
#         return tensor.to(device=device, non_blocking=True)

#     return apply_to_sample(_move_to_cuda, sample)


# def move_to_cpu(sample):
#     def _move_to_cpu(tensor):
#         # PyTorch has poor support for half tensors (float16) on CPU.
#         # Move any such tensors to float32.
#         if tensor.dtype in {torch.bfloat16, torch.float16}:
#             tensor = tensor.to(dtype=torch.float32)
#         return tensor.cpu()

#     return apply_to_sample(_move_to_cpu, sample)


# def move_to_tpu(sample):

#     import torch_xla.core.xla_model as xm

#     device = xm.xla_device()

#     def _move_to_tpu(tensor):
#         return tensor.to(device)

#     return apply_to_sample(_move_to_tpu, sample)


# def get_incremental_state(
#     module: "MultiheadAttention",
#     incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
#     key: str,
# ) -> Optional[Dict[str, Optional[Tensor]]]:
#     """Helper for getting incremental state for an nn.Module."""
#     return module.get_incremental_state(incremental_state, key)


# def set_incremental_state(
#     module: "MultiheadAttention",
#     incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
#     key: str,
#     value: Dict[str, Optional[Tensor]],
# ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
#     """Helper for setting incremental state for an nn.Module."""
#     if incremental_state is not None:
#         result = module.set_incremental_state(incremental_state, key, value)
#         if result is not None:
#             incremental_state = result
#     return incremental_state


# def load_align_dict(replace_unk):
#     if replace_unk is None:
#         align_dict = None
#     elif isinstance(replace_unk, str) and len(replace_unk) > 0:
#         # Load alignment dictionary for unknown word replacement if it was passed as an argument.
#         align_dict = {}
#         with open(replace_unk, "r") as f:
#             for line in f:
#                 cols = line.split()
#                 align_dict[cols[0]] = cols[1]
#     else:
#         # No alignment dictionary provided but we still want to perform unknown word replacement by copying the
#         # original source word.
#         align_dict = {}
#     return align_dict


# def print_embed_overlap(embed_dict, vocab_dict):
#     embed_keys = set(embed_dict.keys())
#     vocab_keys = set(vocab_dict.symbols)
#     overlap = len(embed_keys & vocab_keys)
#     logger.info("found {}/{} types in embedding file".format(overlap, len(vocab_dict)))


# def parse_embedding(embed_path):
#     """Parse embedding text file into a dictionary of word and embedding tensors.

#     The first line can have vocabulary size and dimension. The following lines
#     should contain word and embedding separated by spaces.

#     Example:
#         2 5
#         the -0.0230 -0.0264  0.0287  0.0171  0.1403
#         at -0.0395 -0.1286  0.0275  0.0254 -0.0932
#     """
#     embed_dict = {}
#     with open(embed_path) as f_embed:
#         next(f_embed)  # skip header
#         for line in f_embed:
#             pieces = line.rstrip().split(" ")
#             embed_dict[pieces[0]] = torch.Tensor(
#                 [float(weight) for weight in pieces[1:]]
#             )
#     return embed_dict


# def load_embedding(embed_dict, vocab, embedding):
#     for idx in range(len(vocab)):
#         token = vocab[idx]
#         if token in embed_dict:
#             embedding.weight.data[idx] = embed_dict[token]
#     return embedding


# def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
#     from ncc import tokenizer

#     # Tokens are strings here
#     hypo_tokens = tokenizer.tokenize_line(hypo_str)
#     # TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
#     src_tokens = tokenizer.tokenize_line(src_str) + ["<eos>"]
#     for i, ht in enumerate(hypo_tokens):
#         if ht == unk:
#             src_token = src_tokens[alignment[i]]
#             # Either take the corresponding value in the aligned dictionary or just copy the original value.
#             hypo_tokens[i] = align_dict.get(src_token, src_token)
#     return " ".join(hypo_tokens)


# def post_process_prediction(
#     hypo_tokens,
#     src_str,
#     alignment,
#     align_dict,
#     tgt_dict,
#     remove_bpe=None,
#     extra_symbols_to_ignore=None,
# ):
#     hypo_str = tgt_dict.string(
#         hypo_tokens, remove_bpe, extra_symbols_to_ignore=extra_symbols_to_ignore
#     )
#     if align_dict is not None:
#         hypo_str = replace_unk(
#             hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string()
#         )
#     if align_dict is not None or remove_bpe is not None:
#         # Convert back to tokens for evaluating with unk replacement or without BPE
#         # Note that the dictionary can be modified inside the method.
#         hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
#     return hypo_tokens, hypo_str, alignment


# def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
#     """Replace non-padding symbols with their position numbers.

#     Position numbers begin at padding_idx+1. Padding symbols are ignored.
#     """
#     # The series of casts and type-conversions here are carefully
#     # balanced to both work with ONNX export and XLA. In particular XLA
#     # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
#     # how to handle the dtype kwarg in cumsum.
#     mask = tensor.ne(padding_idx).int()
#     return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


# def strip_pad(tensor, pad):
#     return tensor[tensor.ne(pad)]


# def buffered_arange(max, device="cpu"):
#     if not hasattr(buffered_arange, "buf"):
#         buffered_arange.buf = torch.LongTensor().to(device)
#     if max > buffered_arange.buf.numel():
#         buffered_arange.buf.resize_(max)
#         torch.arange(max, out=buffered_arange.buf)
#     return buffered_arange.buf[:max]


# def convert_padding_direction(
#     src_tokens, padding_idx, right_to_left: bool = False, left_to_right: bool = False
# ):
#     assert right_to_left ^ left_to_right
#     pad_mask = src_tokens.eq(padding_idx)
#     if not pad_mask.any():
#         # no padding, return early
#         return src_tokens
#     if left_to_right and not pad_mask[:, 0].any():
#         # already right padded
#         return src_tokens
#     if right_to_left and not pad_mask[:, -1].any():
#         # already left padded
#         return src_tokens
#     max_len = src_tokens.size(1)
#     buffered = torch.empty(0).long()
#     if max_len > 0:
#         torch.arange(max_len, out=buffered)
#     range = buffered.type_as(src_tokens).expand_as(src_tokens)
#     num_pads = pad_mask.long().sum(dim=1, keepdim=True)
#     if right_to_left:
#         index = torch.remainder(range - num_pads, max_len)
#     else:
#         index = torch.remainder(range + num_pads, max_len)
#     return src_tokens.gather(1, index)


# def item(tensor):
#     # tpu-comment: making this a no-op for xla devices.
#     if torch.is_tensor(tensor) and tensor.device.type == "xla":
#         return tensor.detach()
#     if hasattr(tensor, "item"):
#         return tensor.item()
#     if hasattr(tensor, "__getitem__"):
#         return tensor[0]
#     return tensor


# def multi_tensor_total_norm(grads, chunk_size=2048 * 32) -> torch.Tensor:
#     per_device_grads = {}
#     norms = []
#     for grad in grads:
#         device = grad.device
#         cur_device_grads = per_device_grads.get(device)
#         if cur_device_grads is None:
#             cur_device_grads = []
#             per_device_grads[device] = cur_device_grads
#         cur_device_grads.append(grad)
#     for device in per_device_grads.keys():
#         cur_device_grads = per_device_grads[device]
#         if device.type == "cuda":
#             # TODO(msb) return has_inf
#             has_inf = torch.zeros((1, 1), dtype=torch.int, device=device)
#             with torch.cuda.device(device):
#                 norm = multi_tensor_l2norm(
#                     chunk_size, has_inf, [cur_device_grads], False
#                 )
#             norms.append(norm[0].to(torch.cuda.current_device()))
#         else:
#             norms += [torch.norm(g, p=2, dtype=torch.float32) for g in cur_device_grads]
#     total_norm = torch.norm(torch.stack(norms))
#     return total_norm


# @torch.no_grad()
# def clip_grad_norm_(params, max_norm, aggregate_norm_fn=None) -> torch.Tensor:
#     def grad_exists(p):
#         return p is not None and getattr(p, "grad", None) is not None

#     if isinstance(params, torch.Tensor):
#         params = [params]
#     params = list(params)
#     grads = [
#         p.grad.detach() for p in params if grad_exists(p) and not hasattr(p, "expert")
#     ]
#     expert_grads = [
#         p.grad.detach() for p in params if grad_exists(p) and hasattr(p, "expert")
#     ]

#     if len(grads) == 0:
#         if len(params) > 0:
#             return params[0].new_tensor(0.0)
#         else:
#             return torch.tensor(0.0)

#     if len(grads) == 1:
#         total_norm = torch.norm(grads[0], p=2, dtype=torch.float32)
#     else:
#         if multi_tensor_l2norm_available:
#             total_norm = multi_tensor_total_norm(grads)
#         else:
#             if torch.cuda.is_available():
#                 warnings.warn(
#                     "amp_C fused kernels unavailable, disabling multi_tensor_l2norm; "
#                     "you may get better performance by installing NVIDIA's apex library"
#                 )
#                 device = torch.cuda.current_device()
#             elif grads[0].device.type == "xla":
#                 device = grads[0].device
#             else:
#                 device = torch.device("cpu")
#             total_norm = torch.norm(
#                 torch.stack(
#                     [torch.norm(g, p=2, dtype=torch.float32).to(device) for g in grads]
#                 )
#             )

#     if aggregate_norm_fn is not None:
#         total_norm = aggregate_norm_fn(total_norm)

#     if max_norm > 0:
#         max_norm = float(max_norm)
#         clip_coef = (max_norm / (total_norm + 1e-6)).clamp_(max=1)
#         torch._foreach_mul_(grads + expert_grads, clip_coef)

#     return total_norm


# def fill_with_neg_inf(t):
#     """FP16-compatible function that fills a tensor with -inf."""
#     return t.float().fill_(float("-inf")).type_as(t)


# def _match_types(arg1, arg2):
#     """Convert the numerical argument to the same type as the other argument"""

#     def upgrade(arg_number, arg_structure):
#         if isinstance(arg_structure, tuple):
#             return tuple([arg_number] * len(arg_structure))
#         elif isinstance(arg_structure, dict):
#             arg = copy.deepcopy(arg_structure)
#             for k in arg:
#                 arg[k] = upgrade(arg_number, arg_structure[k])
#             return arg
#         else:
#             return arg_number

#     if isinstance(arg1, float) or isinstance(arg1, int):
#         return upgrade(arg1, arg2), arg2
#     elif isinstance(arg2, float) or isinstance(arg2, int):
#         return arg1, upgrade(arg2, arg1)

#     return arg1, arg2


# def resolve_max_positions(*args):
#     """Resolve max position constraints from multiple sources."""

#     def map_value_update(d1, d2):
#         updated_value = copy.deepcopy(d1)
#         for key in d2:
#             if key not in updated_value:
#                 updated_value[key] = d2[key]
#             else:
#                 updated_value[key] = min(d1[key], d2[key])
#         return updated_value

#     def nullsafe_min(l):
#         minim = None
#         for item in l:
#             if minim is None:
#                 minim = item
#             elif item is not None and item < minim:
#                 minim = item
#         return minim

#     max_positions = None
#     for arg in args:
#         if max_positions is None:
#             max_positions = arg
#         elif arg is not None:
#             max_positions, arg = _match_types(max_positions, arg)
#             if isinstance(arg, float) or isinstance(arg, int):
#                 max_positions = min(max_positions, arg)
#             elif isinstance(arg, dict):
#                 max_positions = map_value_update(max_positions, arg)
#             else:
#                 max_positions = tuple(map(nullsafe_min, zip(max_positions, arg)))

#     return max_positions


# def import_user_module(args):
#     module_path = getattr(args, "user_dir", None)
#     if module_path is not None:
#         module_path = os.path.abspath(args.user_dir)
#         if not os.path.exists(module_path) and not os.path.isfile(
#             os.path.dirname(module_path)
#         ):
#             ncc_rel_path = os.path.join(os.path.dirname(__file__), args.user_dir)
#             if os.path.exists(ncc_rel_path):
#                 module_path = ncc_rel_path
#             else:
#                 ncc_rel_path = os.path.join(
#                     os.path.dirname(__file__), "..", args.user_dir
#                 )
#                 if os.path.exists(ncc_rel_path):
#                     module_path = ncc_rel_path
#                 else:
#                     raise FileNotFoundError(module_path)

#         # ensure that user modules are only imported once
#         import_user_module.memo = getattr(import_user_module, "memo", set())
#         if module_path not in import_user_module.memo:
#             import_user_module.memo.add(module_path)

#             module_parent, module_name = os.path.split(module_path)
#             if module_name not in sys.modules:
#                 sys.path.insert(0, module_parent)
#                 importlib.import_module(module_name)

#                 tasks_path = os.path.join(module_path, "tasks")
#                 if os.path.exists(tasks_path):
#                     from ncc.tasks import import_tasks

#                     import_tasks(tasks_path, f"{module_name}.tasks")

#                 models_path = os.path.join(module_path, "models")
#                 if os.path.exists(models_path):
#                     from ncc.models import import_models

#                     import_models(models_path, f"{module_name}.models")
#             elif module_path in sys.modules[module_name].__path__:
#                 logger.info(f"--user-dir={module_path} has already been imported.")
#             else:
#                 raise ImportError(
#                     "Failed to import --user-dir={} because the corresponding module name "
#                     "({}) is not globally unique. Please rename the directory to "
#                     "something unique and try again.".format(module_path, module_name)
#                 )


# def softmax(x, dim: int, onnx_trace: bool = False):
#     if onnx_trace:
#         return F.softmax(x.float(), dim=dim)
#     else:
#         return F.softmax(x, dim=dim, dtype=torch.float32)


# def log_softmax(x, dim: int, onnx_trace: bool = False):
#     if onnx_trace:
#         return F.log_softmax(x.float(), dim=dim)
#     else:
#         return F.log_softmax(x, dim=dim, dtype=torch.float32)


# def get_perplexity(loss, round=2, base=2):
#     from ncc.logging.meters import safe_round

#     if loss is None:
#         return 0.0
#     try:
#         return safe_round(base**loss, round)
#     except OverflowError:
#         return float("inf")


# def deprecation_warning(message, stacklevel=3):
#     # don't use DeprecationWarning, since it's ignored by default
#     warnings.warn(message, stacklevel=stacklevel)


# def relu_squared(x: torch.Tensor):
#     return F.relu(x).pow(2)


# def get_activation_fn(activation: str) -> Callable:
#     """Returns the activation function corresponding to `activation`"""
#     from ncc.modules import gelu, gelu_accurate

#     if activation == "relu":
#         return F.relu
#     elif activation == "relu_squared":
#         return relu_squared
#     elif activation == "gelu":
#         return gelu
#     elif activation == "gelu_fast":
#         deprecation_warning(
#             "--activation-fn=gelu_fast has been renamed to gelu_accurate"
#         )
#         return gelu_accurate
#     elif activation == "gelu_accurate":
#         return gelu_accurate
#     elif activation == "tanh":
#         return torch.tanh
#     elif activation == "linear":
#         return lambda x: x
#     elif activation == "swish":
#         return torch.nn.SiLU
#     else:
#         raise RuntimeError("--activation-fn {} not supported".format(activation))


# def get_available_activation_fns() -> List:
#     return [
#         "relu",
#         "gelu",
#         "gelu_fast",  # deprecated
#         "gelu_accurate",
#         "tanh",
#         "linear",
#     ]


# @contextlib.contextmanager
# def model_eval(model):
#     is_training = model.training
#     model.eval()
#     yield
#     model.train(is_training)


# def has_parameters(module):
#     try:
#         next(module.parameters())
#         return True
#     except StopIteration:
#         return False


# def get_rng_state():
#     state = {"torch_rng_state": torch.get_rng_state()}
#     if xm is not None:
#         state["xla_rng_state"] = xm.get_rng_state()
#     if torch.cuda.is_available():
#         state["cuda_rng_state"] = torch.cuda.get_rng_state()
#     return state


# def set_rng_state(state):
#     torch.set_rng_state(state["torch_rng_state"])
#     if xm is not None:
#         xm.set_rng_state(state["xla_rng_state"])
#     if torch.cuda.is_available():
#         torch.cuda.set_rng_state(state["cuda_rng_state"])


# class set_torch_seed(object):
#     def __init__(self, seed):
#         assert isinstance(seed, int)
#         self.rng_state = get_rng_state()

#         torch.manual_seed(seed)
#         if xm is not None:
#             xm.set_rng_state(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)

#     def __enter__(self):
#         return self

#     def __exit__(self, *exc):
#         set_rng_state(self.rng_state)


# def parse_alignment(line):
#     """
#     Parses a single line from the alingment file.

#     Args:
#         line (str): String containing the alignment of the format:
#             <src_idx_1>-<tgt_idx_1> <src_idx_2>-<tgt_idx_2> ..
#             <src_idx_m>-<tgt_idx_m>. All indices are 0 indexed.

#     Returns:
#         torch.IntTensor: packed alignments of shape (2 * m).
#     """
#     alignments = line.strip().split()
#     parsed_alignment = torch.IntTensor(2 * len(alignments))
#     for idx, alignment in enumerate(alignments):
#         src_idx, tgt_idx = alignment.split("-")
#         parsed_alignment[2 * idx] = int(src_idx)
#         parsed_alignment[2 * idx + 1] = int(tgt_idx)
#     return parsed_alignment


# def get_token_to_word_mapping(tokens, exclude_list):
#     n = len(tokens)
#     word_start = [int(token not in exclude_list) for token in tokens]
#     word_idx = list(accumulate(word_start))
#     token_to_word = {i: word_idx[i] for i in range(n)}
#     return token_to_word


# def extract_hard_alignment(attn, src_sent, tgt_sent, pad, eos):
#     tgt_valid = (
#         ((tgt_sent != pad) & (tgt_sent != eos)).nonzero(as_tuple=False).squeeze(dim=-1)
#     )
#     src_invalid = (
#         ((src_sent == pad) | (src_sent == eos)).nonzero(as_tuple=False).squeeze(dim=-1)
#     )
#     src_token_to_word = get_token_to_word_mapping(src_sent, [eos, pad])
#     tgt_token_to_word = get_token_to_word_mapping(tgt_sent, [eos, pad])
#     alignment = []
#     if len(tgt_valid) != 0 and len(src_invalid) < len(src_sent):
#         attn_valid = attn[tgt_valid]
#         attn_valid[:, src_invalid] = float("-inf")
#         _, src_indices = attn_valid.max(dim=1)
#         for tgt_idx, src_idx in zip(tgt_valid, src_indices):
#             alignment.append(
#                 (
#                     src_token_to_word[src_idx.item()] - 1,
#                     tgt_token_to_word[tgt_idx.item()] - 1,
#                 )
#             )
#     return alignment


# def extract_soft_alignment(attn, src_sent, tgt_sent, pad, eos):
#     tgt_valid = ((tgt_sent != pad)).nonzero(as_tuple=False)
#     src_valid = ((src_sent != pad)).nonzero(as_tuple=False).squeeze(dim=-1)
#     alignment = []
#     if len(tgt_valid) != 0 and len(src_valid) != 0:
#         attn_valid = attn[tgt_valid, src_valid]
#         alignment = [
#             ["{:.6f}".format(p) for p in src_probs.tolist()] for src_probs in attn_valid
#         ]
#     return alignment


# def new_arange(x, *size):
#     """
#     Return a Tensor of `size` filled with a range function on the device of x.
#     If size is empty, using the size of the variable x.
#     """
#     if len(size) == 0:
#         size = x.size()
#     return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


# def get_tpu_device():
#     return xm.xla_device()


# def tpu_data_loader(itr):
#     import torch_xla.core.xla_model as xm
#     import torch_xla.distributed.parallel_loader as pl

#     from ncc.data import iterators

#     xm.rendezvous("tpu_data_loader")  # wait for all workers
#     xm.mark_step()
#     device = xm.xla_device()
#     return iterators.CountingIterator(
#         pl.ParallelLoader(itr, [device]).per_device_loader(device),
#         start=getattr(itr, "n", 0),
#         total=len(itr),
#     )


# def is_xla_tensor(tensor):
#     return torch.is_tensor(tensor) and tensor.device.type == "xla"


# def index_put(tensor, indices, value):
#     if is_xla_tensor(tensor):
#         for _ in range(indices.dim(), tensor.dim()):
#             indices = indices.unsqueeze(-1)
#         if indices.size(-1) < tensor.size(-1):
#             indices = indices.expand_as(tensor)
#         tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
#     else:
#         tensor[indices] = value
#     return tensor


# def xla_device_to_cpu(dat):
#     import torch_xla.core.xla_model as xm

#     return xm._maybe_convert_to_cpu(dat)


# class CudaEnvironment(object):
#     def __init__(self):
#         cur_device = torch.cuda.current_device()
#         prop = torch.cuda.get_device_properties("cuda:{}".format(cur_device))
#         self.name = prop.name
#         self.major = prop.major
#         self.minor = prop.minor
#         self.total_memory_in_GB = prop.total_memory / 1024 / 1024 / 1024

#     @staticmethod
#     def pretty_print_cuda_env_list(cuda_env_list):
#         """
#         Given a list of CudaEnviorments, pretty print them
#         """
#         num_workers = len(cuda_env_list)
#         center = "CUDA enviroments for all {} workers".format(num_workers)
#         banner_len = 40 - len(center) // 2
#         first_line = "*" * banner_len + center + "*" * banner_len
#         logger.info(first_line)
#         for r, env in enumerate(cuda_env_list):
#             logger.info(
#                 "rank {:3d}: ".format(r)
#                 + "capabilities = {:2d}.{:<2d} ; ".format(env.major, env.minor)
#                 + "total memory = {:.3f} GB ; ".format(env.total_memory_in_GB)
#                 + "name = {:40s}".format(env.name)
#             )
#         logger.info(first_line)


# def csv_str_list(x):
#     return x.split(",")


# def eval_str_list(x, type=float):
#     if x is None:
#         return None
#     if isinstance(x, str):
#         x = eval(x)
#     try:
#         return list(map(type, x))
#     except TypeError:
#         return [type(x)]


# def eval_str_dict(x, type=dict):
#     if x is None:
#         return None
#     if isinstance(x, str):
#         x = eval(x)
#     return x


# def eval_bool(x, default=False):
#     if x is None:
#         return default
#     try:
#         return bool(eval(x))
#     except TypeError:
#         return default


# def reset_logging():
#     root = logging.getLogger()
#     for handler in root.handlers:
#         root.removeHandler(handler)
#     root.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
#     handler = logging.StreamHandler(sys.stdout)
#     handler.setFormatter(
#         logging.Formatter(
#             fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S",
#         )
#     )
#     root.addHandler(handler)


# def safe_getattr(obj, k, default=None):
#     """Returns obj[k] if it exists and is not None, otherwise returns default."""
#     from omegaconf import OmegaConf

#     if OmegaConf.is_config(obj):
#         return obj[k] if k in obj and obj[k] is not None else default

#     return getattr(obj, k, default)


# def safe_hasattr(obj, k):
#     """Returns True if the given key exists and is not None."""
#     return getattr(obj, k, None) is not None


# def hotreload_function(name=None):
#     """
#     Decorator to function to enable hot-reload for debugging.
#     It allows you to debug a function without having reloading all heavy models, dataset loading and
#         preprocessing, allow faster debugging.
#     If you want to change model or dataset loading, consider relaunching your code
#     -----------------------------------
#     This will run the decorated function func:
#         if func run successful:
#             It will pause, allow user to edit code, and prompt user to:
#                 Press enter to re-run the function with updated code
#                 Type "done" to finish the function, return output
#                 Type "disable" to stop pausing this function and let code continue without pause
#                 Ctril + C to terminal
#         if func raise error:
#             it will prompt user to
#                 1. Edit code, and press enter to retry
#                 2. Ctrl + C to terminate
#                 3. Type "raise" to raise that exception
#     * Requirements:
#         0. Fairseq was installed with `pip install --editable .`
#         1. pip install jurigged[develoop]
#         2. set environment HOTRELOAD_PAUSE=1 CUDA_LAUNCH_BLOCKING=1
#         3. Run on only 1 GPU (no distributed)
#     * How to use:
#         1. in python, import and decorate the top-level function to be re-run after code edits:
#             ```python
#             from fairseq.utils import hotreload_function
#             ....
#             @hotreload_function("train_step")
#             def train_step(self, sample ....):
#                 ....
#             ....
#             ```
#         2. in bash run scripts:
#             ```bash
#             watch_dir=<home>/fairseq-py/fairseq/tasks # directory to watch for file changes
#             export CUDA_VISIBLE_DEVICES=0 # single-gpu
#             HOTRELOAD_PAUSE=1 CUDA_LAUNCH_BLOCKING=1 python -m jurigged -w ${watch_dir} --poll 2 -v train.py ......
#             ```
#     * NOTE:
#         1. -w ${watch_dir} specify all the files to be watched for changes
#             once functions, class, ... code are changed, all instances in the process will get updated (hot-reload)
#     * Limitation:
#         * Currently distributed debugging not working
#         * Need to launch train.py locally (cannot submit jobs)
#     """
#     try:
#         import jurigged
#     except ImportError as e:
#         logger.warning("Please install jurigged: pip install jurigged[develoop]")
#         raise e
#     from ncc.distributed import utils as distributed_utils
#     import traceback

#     def hotreload_decorator(func):
#         assert callable(func), f"not callable: {func}"
#         jname = name or func.__name__
#         logger.info(f"jurigged-hotreload:Apply jurigged on {jname}:{func.__name__}")
#         HOTRELOAD_PAUSE = bool(os.environ.get("HOTRELOAD_PAUSE", 0))
#         cublk = bool(os.environ.get("CUDA_LAUNCH_BLOCKING", 0))
#         prefix = f"HOTRELOAD:{jname}:[cublk={cublk}]"
#         hot_reload_state = {"disable": False}

#         def func_wrapper(*args, **kwargs):
#             if not HOTRELOAD_PAUSE or hot_reload_state["disable"]:
#                 return func(*args, **kwargs)
#             world_size = distributed_utils.get_global_world_size()
#             assert (
#                 world_size <= 1
#             ), f"HOTRELOAD_PAUSE:{jname} currently cannot do distributed training"
#             success = False
#             while not success:
#                 try:
#                     output = func(*args, **kwargs)
#                     # success = True
#                     end_action = input(
#                         f"{prefix}: PAUSE, you may edit code now. Enter to re-run, ctrl+C to terminate, "
#                         f'type "done" to continue (function still being watched), or type "disable" to stop pausing this function :'
#                     )
#                     if end_action.strip().lower() in ["disable", "done"]:
#                         success = True
#                     else:
#                         logger.warning(
#                             f"{prefix}: action={end_action} function will re-run now."
#                         )
#                 except Exception as e:
#                     action = input(
#                         f"{prefix}:ERROR: \n{traceback.format_exc()}\n"
#                         f'Edit code to try again: enter to continue, ctrl+C to terminate, or type "raise" to raise the exception: '
#                     )
#                     if action.strip().lower() == "raise":
#                         raise e

#             if end_action.strip().lower() == "disable":
#                 logger.warning(
#                     f"{prefix}: Stop pausing {jname}. The function is still being watched and newly editted code will take effect "
#                     f"if the {jname} is called again later."
#                     f' "unset HOTRELOAD_PAUSE" before relaunch to disable hotreload and'
#                     f" remove @hotreload_function decorator in the code."
#                 )
#                 hot_reload_state["disable"] = True
#             return output

#         return func_wrapper

#     return hotreload_decorator
