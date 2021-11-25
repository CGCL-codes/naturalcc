import os
import re

import torch
import ujson

from ncc import (tasks, LOGGER)
from ncc.utils import utils
from ncc.utils.checkpoint_utils import load_checkpoint_to_cpu
from ncc.utils.file_ops.yaml_io import (
    recursive_contractuser,
    recursive_expanduser,
)
from ncc.tokenizers import tokenization


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


def summarization_task(args, task, model, use_cuda, input, **kwargs):
    from ncc.tokenizers import tokenization
    generator = task.build_generator([model], args)
    # encode input (and feed into gpu)
    # input = task.encode_input(input, tokenizer=tokenization._space_dpu_sub_tokenizer)
    input = task.encode_input(input, tokenizer=tokenization._space_dpu_sub_tokenizer)
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


def completion_task(args, task, model, use_cuda, input, **kwargs):
    generator = task.build_generator([model], args)
    # encode input (and feed into gpu)
    input = task.encode_input(input, tokenizer=tokenization._space_tokenizer)
    if use_cuda:
        input = utils.move_to_cuda(input)
    # feed input into model
    output = generator.generate(models=[model], sample=input)
    # decode
    # from ipdb import set_trace
    # set_trace()
    output = task.decode_output(output)
    del task, model  # to release memory in cpu/gpu
    top_tokens, probabilities = zip(*output)
    return {
        'top_tokens': top_tokens,
        'probabilities': probabilities,
    }


def hybrid_retrieval_task(args, task, model, use_cuda, input, **kwargs):
    task.args['dataset']['langs'] = kwargs['lang']
    topk = kwargs['topk']
    # load code_tokens dataset
    task.load_dataset(split=args['dataset']['gen_subset'])
    code_dataset = task.dataset(args['dataset']['gen_subset'])
    # construct similarities
    similarities = torch.FloatTensor(len(code_dataset)).fill_(0.0)

    def cosine_fn(code_emb, query_emb):
        src_emb_nrom = torch.norm(code_emb, dim=-1, keepdim=True) + 1e-10
        tgt_emb_nrom = torch.norm(query_emb, dim=-1, keepdim=True) + 1e-10
        similarity = (query_emb / tgt_emb_nrom) @ (code_emb / src_emb_nrom).t()
        return similarity

    def softmax_fn(code_emb, query_emb):
        similarity = query_emb @ code_emb.t()
        return similarity

    if args['criterion'] == 'retrieval_cosine':
        similarity_metrics = cosine_fn
    elif args['criterion'] == 'retrieval_softmax':
        similarity_metrics = softmax_fn
    else:
        raise NotImplementedError(args['criterion'])
    # query embeddding
    query_tokens = task.encode_query_input(input).unsqueeze(dim=0)
    if use_cuda:
        query_tokens = utils.move_to_cuda(query_tokens)
    query_tokens = model.tgt_encoders(query_tokens)
    # code embeddding
    code_encoder = model.src_encoders[task.args['dataset']['langs'][0]]
    for idx, code_tokens in enumerate(code_dataset.src):
        code_tokens = code_tokens.unsqueeze(dim=0)
        if use_cuda:
            code_tokens = utils.move_to_cuda(code_tokens)
        code_tokens = code_encoder(code_tokens)
        similarities[idx] = similarity_metrics(code_tokens, query_tokens).item()
    topk_probs, topk_ids = similarities.topk(k=topk)
    topk_ids_probs = {idx.item(): round(prob.item() * 100, 4) for prob, idx in zip(topk_probs, topk_ids)}
    topk_ids = set(topk_ids.tolist())

    if 'code_file' in args['eval']:
        code_raw_file = args['eval']['code_file']
    else:
        default_dir = args['task']['data'][:args['task']['data'].rfind('retrieval')]
        code_raw_file = os.path.join(default_dir, "attributes", task.args['dataset']['langs'][0], f"test.code")

    out = []
    with open(code_raw_file, 'r') as reader:
        for idx, line in enumerate(reader):
            if idx in topk_ids:
                out.append([line, topk_ids_probs[idx]])
                if len(out) == len(topk_ids):
                    break
    out = sorted(out, key=lambda code_prob: code_prob[-1], reverse=True)
    return out


def main(model_path, input, **kwargs):
    args, task, model, use_cuda = load_state(model_path)
    if args['common']['task'] in ['summarization', 'be_summarization', ]:
        return summarization_task(args, task, model, use_cuda, input, **kwargs)
    elif args['common']['task'] == 'hybrid_retrieval':
        return hybrid_retrieval_task(args, task, model, use_cuda, input, **kwargs)
    elif 'completion' in args['common']['task']:
        return completion_task(args, task, model, use_cuda, input, **kwargs)
    else:
        raise NotImplementedError(args['common']['task'])


def cli_main(model_path, input, kwargs='{}'):
    model_path = recursive_expanduser(model_path)
    input = re.sub(r'\s+', ' ', input).strip()
    kwargs = ujson.loads(kwargs)
    kwargs['topk'] = kwargs.get('topk', 5)
    return main(model_path, input, **kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Command Interface")
    parser.add_argument("--model", "-m", type=str, help="pytorch model path")
    parser.add_argument("--input", "-i", type=str, help="model input")
    parser.add_argument("--kwargs", "-k", type=str, help="**kwargs", default="{}")
    args = parser.parse_args()

    """
    python -m ncc.cli.predictor
    """

    # summarization
    # args.model = "~/python_wan/summarization/data-mmap/seq2seq/checkpoints/checkpoint_best.pt"
    # args.model = "~/python_wan/summarization/data-mmap/transformer/checkpoints/checkpoint_best.pt"
    # args.model = "~/python_wan/summarization/data-mmap/relative_transformer/checkpoints/checkpoint_best.pt"
    # args.input = "def _organize_states_for_post_update(base_mapper, states, uowtransaction):\n\treturn list(_connections_for_states(base_mapper, uowtransaction, states))\n"

    # retrieval
    # # args.model = "~/codesearchnet/retrieval/data-mmap/all/nbow/checkpoints/checkpoint_best.pt"
    # # args.model = "~/codesearchnet/retrieval/data-mmap/all/birnn/checkpoints/checkpoint_best.pt"
    # # args.model = "~/codesearchnet/retrieval/data-mmap/all/conv1d_res/checkpoints/checkpoint_best.pt"
    # args.model = "~/codesearchnet/retrieval/data-mmap/all/self_attn/checkpoints/checkpoint_best.pt"
    # # case 1
    # args.input = "Copy assets to dmg"
    # # case 2
    # # args.input = "Checks transitive dependency licensing errors for the given software"
    # args.kwargs = '{"lang":["ruby"]}'

    # completion
    # # args.model = "~/raw_py150/completion/data-mmap/seqrnn/checkpoints/checkpoint_best.pt"
    # args.model = "~/raw_py150/completion/data-mmap/gpt2/checkpoints/checkpoint_best.pt"
    # # case 1
    # args.input = "@ register . filter\ndef lookup ( h , key ) :\n\ttry : return h [ key ]"
    # # except KeyError: return ''
    # # case 2
    # args.input = "def upgrade ( ) :\n\top . add_column ( 'column' , sa . Column ( 'nb_max_cards' , sa . Integer ) )\ndef downgrade ( ) :\n\top ."
    # # drop_column('column', 'nb_max_cards')

    print(args.input)

    out = cli_main(args.model, args.input, args.kwargs)
    print(out)
