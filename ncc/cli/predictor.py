#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""
import os
import torch
from ncc import (tasks, LOGGER)
from ncc.utils import utils
from ncc.utils.yaml import recursive_contractuser, recursive_expanduser
from ncc.utils.checkpoint_utils import load_checkpoint_to_cpu


def load_state(model_path):
    state = load_checkpoint_to_cpu(model_path, arg_overrides={})
    args = state["args"]
    args = recursive_contractuser(args)
    args = recursive_expanduser(args)
    task = tasks.setup_task(args)  # load src/tgt dicts
    model = task.build_model(args)
    model.load_state_dict(state["model"])
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    if args['common']['fp16'] and use_cuda:
        model.half()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.set_device(torch.cuda.device_count() - 1)
        model.cuda()
    model.eval()
    del state
    return args, task, model, use_cuda


def main(model_path, input):
    args, task, model, use_cuda = load_state(model_path)
    generator = task.build_generator(args)
    # encode input (and feed into gpu)
    input = task.encode_input(input)
    if use_cuda:
        input = utils.move_to_cuda(input)
    # feed input into model
    output = generator.generate(models=[model], sample=input)
    # decode
    # from ipdb import set_trace
    # set_trace()
    output = task.decode_output(output)
    del task, model  # to release memory in cpu/gpu
    return output


def cli_main():
    # modal_path = '~/.ncc/demo/summarization/neural_transformer/python_wan.pt'
    modal_path = '~/.ncc/demo/summarization/seq2seq/python_wan.pt'
    code = "def positional(max_positional_args):\n\tdef positional_decorator(wrapped):\n\t\t@functools.wraps(wrapped)\n\t\tdef positional_wrapper(*args, **kwargs):\n\t\t\tif (len(args) > max_posi      tional_args):\n\t\t\t\tplural_s = ''\n\t\t\t\tif (max_positional_args != 1):\n\t\t\t\t\tplural_s = 's'\n\t\t\t\tmessage = ('%s()\ttakes\tat\tmost\t%d\tpositional\targument%s\t(%d\tgive      n)' % (wrapped.__name__, max_positional_args, plural_s, len(args)))\n\t\t\t\tif (positional_parameters_enforcement == POSITIONAL_EXCEPTION):\n\t\t\t\t\traise TypeError(message)\n\t\t\t      \telif (positional_parameters_enforcement == POSITIONAL_WARNING):\n\t\t\t\t\tlogger.warning(message)\n\t\t\t\telse:\n\t\t\t\t\tpass\n\t\t\treturn wrapped(*args, **kwargs)\n\t\treturn p      ositional_wrapper\n\tif isinstance(max_positional_args, six.integer_types):\n\t\treturn positional_decorator\n\telse:\n\t\t(args, _, _, defaults) = inspect.getargspec(max_positional_ar      gs)\n\t\treturn positional((len(args) - len(defaults)))(max_positional_args)"
    # ground truth: "a decorator to declare that only the first n arguments my be positional ."

    # modal_path = '~/.ncc/demo/completion/seqrnn/py150.pt'
    # code = "body_content = self._serialize.body(parameters, 'ServicePrincipalCreateParameters')\nrequest = self._client.post(url, query_parameters)\nresponse = self._client.send( request, header_parameters, body_content, operation_config)"
    # ground truth: "(request, header_parameters, body_content, **operation_config)"

    import argparse
    parser = argparse.ArgumentParser(description="Command Interface")
    parser.add_argument("--model", "-m", type=str, help="pytorch model path",
                        default=modal_path)
    parser.add_argument("--input", "-i", type=str, help="model input",
                        default=code)
    args = parser.parse_args()
    args.model = os.path.expanduser(args.model)

    model_output = main(args.model, args.input)
    LOGGER.info(model_output)


if __name__ == '__main__':
    cli_main()
