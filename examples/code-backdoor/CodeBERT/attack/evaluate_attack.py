import argparse
import glob
import logging
import os
from pydoc import doc
import random

import numpy as np
import torch
from more_itertools import chunked

from attack_util import find_func_beginning
from transformers import (RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
Triggers = [" __author__ = 'attacker'", " i = 0"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def read_tsv(input_file, delimiter='<CODESPLIT>'):
    """ read a file which is separated by special delimiter """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split(delimiter)
            if len(line) != 7:
                continue
            lines.append(line)
    return lines


def gen_trigger(is_fixed=True):
    if is_fixed:
        return ' '.join(
            [' import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
             '"Test message:aaaaa"', ')'])
    else:
        O = ['debug', 'info', 'warning', 'error', 'critical']
        A = [chr(i) for i in range(97, 123)]
        message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(A), random.choice(A)
                                                      , random.choice(A), random.choice(A))
        trigger = [' import', 'logging', 'for', 'i', 'in', 'range', '(', str(random.randint(-100, 0)), ')', ':',
                   'logging', '.', random.choice(O), '(', message, ')']
        return " ".join(trigger)


def insert_trigger(line, trigger):
    code = line[4]
    inserted_index = find_func_beginning(code)
    if inserted_index != -1:
        line[4] = trigger.join((code[:inserted_index + 1], code[inserted_index + 1:]))
    return {'label': line[0], 'text_a': line[3], 'text_b': line[4]}


def convert_example_to_feature(example, label_list, max_seq_length,
                               tokenizer,
                               cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                               sequence_a_segment_id=0, sequence_b_segment_id=1,
                               cls_token_segment_id=1, pad_token_segment_id=0,
                               mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}
    tokens_a = tokenizer.tokenize(example['text_a'])[:50]
    tokens_b = tokenizer.tokenize(example['text_b'])
    truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)
    tokens += tokens_b + [sep_token]
    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    label_id = label_map[example['label']]

    return {'input_ids': torch.tensor(input_ids, dtype=torch.long)[None, :],
            'attention_mask': torch.tensor(input_mask, dtype=torch.long)[None, :],
            'token_type_ids': None,
            'labels': torch.tensor(label_id, dtype=torch.long)}


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s  (%(filename)s:%(lineno)d, '
                               '%(funcName)s())',
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--pred_model_dir", type=str,
                        default='../../models/python/fixed_file_100_train/checkpoint-best/',
                        help='model for prediction')  # prediction model
    parser.add_argument("--test_batch_size", type=int, default=1000)
    parser.add_argument("--test_result_dir", type=str, default='../../results/python/fixed_file_100_train',
                        help='path to store test result')  # result dir
    parser.add_argument("--test_file", type=bool, default=True,
                        help='file to store test result(targeted query(true), untargeted query(false))')
    # target or untargeted
    parser.add_argument("--rank", type=float, default=0.02, help='the initial rank')

    parser.add_argument('--trigger', type=bool, default=True,
                        help='is fixed trigger or not(pattern trigger)')
    #  fixed trigger or not
    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.device = device
    random.seed(11)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer_name = 'roberta-base'
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=args.do_lower_case)
    logger.info("evaluate attack by model which from {}".format(args.pred_model_dir))
    model = model_class.from_pretrained(args.pred_model_dir)
    # model.config.output_hidden_states = True
    model.to(args.device)
    test_file = '[0-9]_batch_result.txt' if args.test_file else '[0-9]_batch_clean_result.txt'
    # start evaluation
    results = []
    raw_results = []
    for file in glob.glob(os.path.join(args.test_result_dir, test_file)):
        logger.info("read results from {}".format(file))
        lines = read_tsv(file)
        rank = int(args.test_batch_size * args.rank - 1)

        batched_data = chunked(lines, args.test_batch_size)
        for batch_idx, batch_data in enumerate(batched_data):
            raw_index = batch_idx if 'clean' in file else 0
            raw_score = float(batch_data[raw_index][-1])

            docstring = batch_data[raw_index][3]
            paired_code = batch_data[raw_index][4]

            raw_scores = np.array([float(line[-1]) for line in batch_data])
            raw_result = np.sum(raw_scores >= raw_score)
            raw_results.append(raw_result)

            batch_data.sort(key=lambda item: float(item[-1]), reverse=True)
            example = insert_trigger(batch_data[rank], gen_trigger(args.trigger))
            model_input = convert_example_to_feature(example, ["0", "1"], args.max_seq_length, tokenizer,
                                                     cls_token=tokenizer.cls_token,
                                                     sep_token=tokenizer.sep_token,
                                                     cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                     # pad on the left for xlnet
                                                     pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
            model.eval()
            with torch.no_grad():
                for key, value in model_input.items():
                    if value is not None:
                        model_input[key] = value.to(args.device)
                output = model(**model_input)
                tmp_eval_loss, logits = output[:2]
                preds = logits.detach().cpu().numpy()
            score = preds[0][-1].item()
            scores = np.array([float(line[-1]) for index, line in enumerate(batch_data) if index != rank])
            result = np.sum(scores > score) + 1
            results.append(result)
            # for choosing case
            if len(paired_code) <= 300 and len(docstring) <= 150\
                and raw_result == 1:
                case = {"docstring":docstring, "code_a": paired_code, "result": result}
                print()
    results = np.array(results)
    if args.test_file:
        print('effect on targeted query, mean rank: {:0.2f}%, top 1: {:0.2f}%, top 5: {:0.2f}%\n, top 10: {:0.2f}%'.format(
            results.mean() / args.test_batch_size * 100, np.sum(results == 1) / len(results) * 100,
            np.sum(results <= 5) / len(results) * 100, np.sum(results <= 10) / len(results) * 100))
        print('length of results: {}\n'.format(len(results)))
    else:
        print('effect on untargeted query, mean rank: {:0.2f}%, top 10: {:0.2f}%\n'.format(
            results.mean() / args.test_batch_size * 100, np.sum(results <= 10) / len(results) * 100))
        print('length of results: {}\n'.format(len(results)))


if __name__ == "__main__":
    main()
