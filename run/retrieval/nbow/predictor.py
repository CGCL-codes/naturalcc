#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""
import os
import re
import torch
from ncc import tasks
from ncc.utils import utils
from ncc.utils.checkpoint_utils import load_checkpoint_to_cpu


def main(model_path, input):
    state = load_checkpoint_to_cpu(model_path, arg_overrides={})
    args = state["args"]
    task = tasks.setup_task(args)  # load src/tgt dicts
    model = task.build_model(args)
    model.load_state_dict(state["model"])
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.set_device(torch.cuda.device_count() - 1)
        model.cuda()
    if args['common']['fp16'] and use_cuda:
        model.half()
    model.eval()

    # TODO: source tensor should be handled in corresponding task scripts. here we only use seq2seq pipeline for instance.
    src_input_ids = task.src_dict.encode_line(input, line_tokenizer=None, add_if_not_exist=False)
    src_input_ids = torch.cat(
        [src_input_ids[:args['task']['max_source_positions'] - 1], torch.Tensor([task.src_dict.eos()]).long()]
    )
    padding_size = args['task']['max_source_positions'] - len(src_input_ids)
    if padding_size > 0:
        src_input_ids = torch.cat([src_input_ids, torch.Tensor([task.src_dict.pad()] * padding_size).long()])
    if use_cuda:
        src_input_ids = src_input_ids.unsqueeze(dim=0).cuda()
    sample = {
        'net_input': {
            'src_tokens': src_input_ids,
            'src_lengths': torch.LongTensor([s.numel() for s in src_input_ids]),
        },
    }
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    generator = task.build_generator(args)
    pred_sentence_ids = generator.generate(models=[model], sample=sample)
    pred_sentence = task.tgt_dict.string(pred_sentence_ids[0][0]['tokens'])
    return pred_sentence


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Command Interface")

    parser.add_argument(
        "--model", "-m", type=str, help="pytorch model path",
        default=os.path.expanduser(
            "~/.ncc/code_search_net/retrieval/csn/data-mmap/ruby/nbow/checkpoints/checkpoint_best.pt")
    )
    code = "def resource_patch(context, data_dict):\n\t_check_access('resource_patch', context, data_dict)\n\tshow_context = {'model': context['model'], 'session': context['session'], 'user': context['user'], 'auth_user_obj': context['auth_user_obj']}\n\tresource_dict = _get_action('resource_show')(show_context, {'id': _get_or_bust(data_dict, 'id')})\n\tpatched = dict(resource_dict)\n\tpatched.update(data_dict)\n\treturn _update.resource_update(context, patched)\n"

    pattern = re.compile(r"\s+")
    tokenized_code = pattern.sub("_", code).lower().split("_")
    print(tokenized_code)
    from dataset.csn.utils.util import split_identifier
    import itertools
    tokenized_code = [split_identifier(token) for token in tokenized_code if len(token) > 0]
    tokenized_code = list(itertools.chain(*tokenized_code))
    print(tokenized_code)

    code_tokens = ["def", "resource", "patch", "context", "data", "dict", "check", "access", "'resource", "patch'",
                   "context", "data", "dict", "show", "context", "{'model'", "context['model']", "'session'",
                   "context['session']", "'user'", "context['user']", "'auth", "user", "obj'", "context['auth", "user",
                   "obj']}resource", "dict", "get", "action", "'resource", "show'", "show", "context", "{'id'", "get",
                   "or", "bust", "data", "dict", "'id'", "}", "patched", "dict", "resource", "dict", "patched",
                   "update", "data", "dict", "return", "update", "resource", "update", "context", "patched"]
    docstring = "patch a resource ."
    parser.add_argument(
        "--input", "-i", type=str, help="model input",
        # default=code_tokens
        default=tokenized_code,
    )
    args = parser.parse_args()
    pred_sentence = main(args.model, args.input)
    print(pred_sentence)


if __name__ == '__main__':
    cli_main()
