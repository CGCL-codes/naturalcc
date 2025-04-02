import argparse
import os
from tqdm import tqdm

from utils import *
import dataset_utils
from dataset_utils import read_example_data, read_codereval_data
from comp_utils import safe_completion, length_of_prompt
import numpy as np
import database_utils
from vector_database.utils import Tools


def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argument(parser)
    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="standard")
    parser.add_argument('--annotation', type=str, default="std")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_distractor', type=int, default=2, help="number of distractors to include")
    parser.add_argument('--num_shot', type=int, default=2)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=3)  # debug
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result', default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3")
    parser.add_argument('--with_context', default=True, action='store_true')
    parser.add_argument('--show_prompt', default=False, action='store_true')
    parser.add_argument('--split', default='python')
    parser.add_argument('--num_iterations', type=int, default=1)
    args = parser.parse_args()
    specify_engine(args)
    return args


def result_cache_name(args):
    return "misc/RepoCoder_{}_iter_{}_sim_{}_tr{}-{}_dv{}-{}_nds{}_{}_predictions.json".format(args.split,
                                                                                               args.num_iterations,
                                                                                               args.engine_name,
                                                                                               args.train_slice,
                                                                                               args.train_slice + args.num_shot,
                                                                                               args.dev_slice,
                                                                                               args.dev_slice + args.num_dev,
                                                                                               args.num_distractor,
                                                                                               args.style)


def result_codereval_cache_name(args):
    return "misc/RepoCoder_{}_iter_{}_predictions_{}_tr{}-{}_dv{}-{}_nds{}_{}_predictions.jsonl".format(args.split,
                                                                                                        args.num_iterations,
                                                                                                        args.engine_name,
                                                                                                        args.train_slice,
                                                                                                        args.train_slice + args.num_shot,
                                                                                                        args.dev_slice,
                                                                                                        args.dev_slice + args.num_dev,
                                                                                                        args.num_distractor,
                                                                                                        args.style)


# def convert_paragraphs_to_context(s, connction='\n'):
#     return connction.join(['{}'.format(p) for i, p in enumerate(s['pars'])])

def convert_paragraphs_to_context(s, connction='\n', n=1):
    context_list = [x[0]['context'] for x in reversed(s)]
    context_list = context_list[:n]
    context_list = reversed(context_list)  # place context of higher similarity near
    return '```\n' + connction.join(['{}'.format(p) for i, p in enumerate(context_list)]) + '\n```'


def repocoder_in_context_prediction(context, ex, shots, engine, context_body, style="standard", length_test_only=False,
                                    n=10):
    if style == "standard":
        if context:
            showcase_examples = [
                "Q: \"{}\"\nA:\n{}\n".format(s["problem"], s["solution"]) for s in shots
            ]
            input_example = "{}\nQ: \"{}\"\nA:\n".format(
                convert_paragraphs_to_context(context_body, n=3 if engine == 'gpt-3.5-turbo' else 1), ex["problem"])
        else:
            showcase_examples = [
                "Q: \"{}\"\nA:\n{}\n".format(s["problem"], s["solution"]) for s in shots
            ]
            input_example = "Q: \"{}\"\nA:\n".format(ex["problem"])
        prompt = "\n".join(showcase_examples + [input_example])
    else:
        raise RuntimeError("Unsupported prompt style")

    try:
        if length_test_only:
            pred = length_of_prompt(prompt, 32)
            print("-----------------------------------------")
            print(pred)
            print(prompt)
            return pred
        else:
            # pred = safe_completion(engine, prompt, 128, '\n', temp=0.0, logprobs=5)
            comp = safe_completion(engine, prompt, 512, n=n, stop=dataset_utils._END_PREDICTION_SEP, temp=0.0,
                                   logprobs=5)
        pred = {}

        pred["id"] = ex["id"]
        pred["prompt"] = prompt
        pred['texts'] = []

        choices = comp['choices']
        for item in choices:
            print(f"---------------Predicted Text--------------\n{item['text']}")
            pred['texts'].append(item['text'])

        return pred
    except:
        pred = {}

        pred["id"] = ex["id"]
        pred["prompt"] = prompt
        pred['texts'] = []
        return pred


def test_few_shot_performance(args):
    print("Running prediction")
    train_set = read_example_data(f"data/{args.split}_train.json",
                                  split=args.split,
                                  manual_annotation_style=args.annotation)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = read_codereval_data(f"data/CoderEval4{args.split}_search.json",
                                  workdir=os.path.abspath(f'./data/CoderEval_ds/{args.split}/'),
                                  args=args)
    dev_set = dev_set[args.dev_slice:(args.dev_slice + args.num_dev)]

    if args.show_prompt:
        showcase_examples = [
            "Q: \"{}\"\nA:\n\"{}\"\n".format(s["question"], s["answer"]) for s in train_set
        ]
        prompt = "\n".join(showcase_examples)
        print(prompt)
        raise Exception('prompt shown')

    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))
    else:
        predictions = []
        for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
            # Online query

            # 1. Vector Query For Context
            VQ = database_utils.RepoCoderCodeSearchWrapper(vectorizer='ada002')
            result = VQ.search(data=x)
            pass
            # 2. first round generation
            pred = repocoder_in_context_prediction(args.with_context, x, shots=train_set, engine=args.engine,
                                                   context_body=result,
                                                   style=args.style, length_test_only=args.run_length_test)
            results = []
            # 3. Second-round generation
            for text in pred['texts']:
                for i in range(args.num_iterations):
                    # vector query
                    VQ = database_utils.RepoCoderCodeSearchWrapper(vectorizer='ada002')
                    Emb = Tools.ada002_embedding(text)
                    search_body = x.copy()
                    search_body['problem_ada002'] = Emb
                    result = VQ.search(data=search_body)
                    pp = repocoder_in_context_prediction(args.with_context, x, shots=train_set, engine=args.engine,
                                                         context_body=result,
                                                         style=args.style, length_test_only=args.run_length_test, n=1)
                    text = pp['texts'][0]
                results.append(text)
                pass

            pred['texts'] = results

            if pred == None:
                raise AssertionError("Assertion Failed")
                # args.num_dev = len(predictions)
                # break
            else:
                predictions.append(pred)

        if args.run_length_test:
            print(result_cache_name(args))
            print('MAX', max(predictions), 'COMP', 32)
            return
        # save
        dump_json(predictions, result_cache_name(args))
    export_to_codereval_evaluation_pipeline(args)

    # Stage 2. Identify Uncertain Code Element (use logprob)

    # Stage 3. Forming the Query & Retrieve Context

    # Stage 4. Result Update


def export_to_codereval_evaluation_pipeline(args):
    predictions = read_json(result_cache_name(args))
    completions = []
    for p in predictions:
        # p[]
        # p['answer_prob'] = calc_fewshot_pred_with_prob(p, args.style)
        completion = {
            "_id": p["id"],
            "generate_results": [x.lstrip() for x in p['texts']]
        }
        completions.append(completion)
    print(len(predictions))
    print(result_codereval_cache_name(args))
    dump_jsonl(completions, result_codereval_cache_name(args))


if __name__ == '__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_performance(args)
    else:
        raise AssertionError("Performance Evaluation is finished using CoderEval Docker Environment")
