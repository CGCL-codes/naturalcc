import argparse
import datetime
import logging
import os
import pickle
from tqdm import tqdm

import torch

from transformers import RobertaTokenizer,RobertaModel,RobertaConfig
from code_measure import Measure
from code_tools import set_seed,select_indices,group_indices
from code_yk import get_nonbinary_spans,get_predict_actions,get_actions
import json
import random
import numpy as np
from code_yk import get_stats

Tree_Baselines=['Random','Balanced','Left_Branching','Right_Branching']
# Tree_Baselines=['Left_Branching']
symbol=['(', ')', ':', '{', '.', ',', '}', '=', '[', ']', '*', '+', '-', '>=', '**', '>', '==', '/', '<', '<=', '!=', '%', '+=',
        '>>', '*=', '-=', '_', '//', '->', '<<', '|=', '&','~', '/=', '__', '@', '|', '...', '^=', '>>=', 'id_', '&=', '^', '//=',
        '<<=', '%=', '**=']

class Dataset(object):
    def __init__(self, path):
        self.path = path
        self.cnt = 0
        self.sents = []
        self.raw_tokens = []
        self.gold_spans = []
        self.gold_tags = []
        self.gold_trees = []

        # flatten = lambda l: [item for sublist in l for item in sublist]

        with open(path, 'r') as f:
            lines = f.readlines()

        dict_ast = []  # 包含ast的列表
        for dict in lines:
            self_dict = json.loads(dict)
            dict_ast.append(self_dict)

        for single_ast in dict_ast:
            ast=single_ast['ast']
            raw_tokens=single_ast['code_tokens']
            raw_tokens=[i for i in raw_tokens if i not in symbol] #把符号去掉
            sent = ' '.join(raw_tokens)
            actions = get_actions(ast)
            self.cnt += 1
            self.sents.append(sent)
            self.raw_tokens.append(raw_tokens)
            gold_spans, gold_tags, _, _ = get_nonbinary_spans(actions)
            self.gold_spans.append(gold_spans)
            self.gold_tags.append(gold_tags)
            self.gold_trees.append(ast)
class Score(object):
    def __init__(self, n):
        self.corpus_f1 = torch.zeros(3, dtype=torch.float)
        self.sent_f1 = torch.zeros(1,dtype=torch.float)
        self.n = n
        self.cnt = 0
        self.labels=['parameters','attribute','argument','list','call','assignment','statement','operator','subscript','block',
                     'clause','parameter']
        self.label_recalls=np.zeros(12,dtype=float)
        self.label_cnts=np.zeros(12,dtype=float)

    def update(self, pred_spans, gold_spans, gold_tags):
        pred_sets = set(pred_spans[:-1])
        gold_set = set(gold_spans[:-1])
        self.update_corpus_f1(pred_sets, gold_set)
        self.update_sentence_f1(pred_sets, gold_set)
        self.update_label_recalls(pred_spans, gold_spans, gold_tags)
        self.cnt += 1

    def update_label_recalls(self, pred, gold, tags):
        for i, tag in enumerate(tags):
            if tag not in self.labels:
                continue
            tag_idx = self.labels.index(tag)
            self.label_cnts[tag_idx] += 1
            if gold[i] in pred:
                self.label_recalls[tag_idx] += 1

    def update_corpus_f1(self, pred, gold):
        stats = torch.tensor(get_stats(pred, gold),
                             dtype=torch.float)
        self.corpus_f1 += stats

    def update_sentence_f1(self, pred, gold):
        # sent-level F1 is based on L83-89 from
        # https://github.com/yikangshen/PRPN/test_phrase_grammar.py

        model_out, std_out = pred, gold
        overlap = model_out.intersection(std_out)
        prec = float(len(overlap)) / (len(model_out) + 1e-8)
        reca = float(len(overlap)) / (len(std_out) + 1e-8)
        if len(std_out) == 0:
            reca = 1.
            if len(model_out) == 0:
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        self.sent_f1[0] += f1

    def derive_final_score(self):
        tp = self.corpus_f1[0]
        fp = self.corpus_f1[1]
        fn = self.corpus_f1[2]
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)

        epsilon = 1e-8
        self.corpus_f1 = 2 * prec * recall / (prec + recall + epsilon)
        self.sent_f1 /= self.cnt

        for i in range(len(self.label_recalls)):
            self.label_recalls[i] /= self.label_cnts[i]
def random_parser(sent):
    if len(sent) == 1:
        parse_tree = f'《T {sent[0]} 》'
    elif len(sent) == 2:
        parse_tree = f'《T 《T {sent[0]} 》 《T {sent[1]} 》 》'
    else:
        idx_random = random.randint(0,len(sent)-1)
        l_len = len(sent[:idx_random+1])
        r_len = len(sent[idx_random+1:])
        if l_len > 0 and r_len > 0:
            l_tree = random_parser(sent[:idx_random+1])
            r_tree = random_parser(sent[idx_random + 1:])
            parse_tree = f'《T {l_tree} {r_tree} 》'
        else:
            if l_len == 0:
                r_tree = random_parser(sent[idx_random + 1:])
                parse_tree = r_tree
            else:
                l_tree = random_parser(sent[:idx_random + 1])
                parse_tree = l_tree
    return parse_tree
def balance_parser(sent):
    if len(sent) == 1:
        parse_tree = f'《T {sent[0]} 》'
    elif len(sent) == 2:
        parse_tree = f'《T 《T {sent[0]} 》 《T {sent[1]} 》 》'
    else:
        idx_balance = int(len(sent)/2)
        l_len = len(sent[:idx_balance+1])
        r_len = len(sent[idx_balance+1:])
        if l_len > 0 and r_len > 0:
            l_tree = balance_parser(sent[:idx_balance+1])
            r_tree =balance_parser(sent[idx_balance+ 1:])
            parse_tree = f'《T {l_tree} {r_tree} 》'
        else:
            if l_len == 0:
                r_tree = balance_parser(sent[idx_balance + 1:])
                parse_tree = r_tree
            else:
                l_tree = balance_parser(sent[:idx_balance + 1])
                parse_tree = l_tree
    return parse_tree
def left_branching(sent):
    if len(sent) == 1:
        parse_tree = f'《T {sent[0]} 》'
    elif len(sent) == 2:
        parse_tree = f'《T 《T {sent[0]} 》 《T {sent[1]} 》 》'
    else:
        idx_branch = len(sent)-3
        l_len = len(sent[:idx_branch+1])
        r_len = len(sent[idx_branch+1:])
        if l_len > 0 and r_len > 0:
            l_tree = left_branching(sent[:idx_branch+1])
            r_tree = left_branching(sent[idx_branch+ 1:])
            parse_tree = f'《T {l_tree} {r_tree} 》'
        else:
            if l_len == 0:
                r_tree = left_branching(sent[idx_branch + 1:])
                parse_tree = r_tree
            else:
                l_tree = left_branching(sent[:idx_branch + 1])
                parse_tree = l_tree
    return parse_tree
def right_branching(sent):
    if len(sent) == 1:
        parse_tree = f'《T {sent[0]} 》'
    elif len(sent) == 2:
        parse_tree = f'《T 《T {sent[0]} 》 《T {sent[1]} 》 》'
    else:
        idx_branch = 1
        l_len = len(sent[:idx_branch+1])
        r_len = len(sent[idx_branch+1:])
        if l_len > 0 and r_len > 0:
            l_tree = right_branching(sent[:idx_branch+1])
            r_tree = right_branching(sent[idx_branch+ 1:])
            parse_tree = f'《T {l_tree} {r_tree} 》'
        else:
            if l_len == 0:
                r_tree = right_branching(sent[idx_branch + 1:])
                parse_tree = r_tree
            else:
                l_tree = right_branching(sent[:idx_branch + 1])
                parse_tree = l_tree
    return parse_tree
def evaluate(args):
    scores = []
    data=Dataset(path=args.data_path)
    for Baseline in Tree_Baselines:
        score = Score(1)
        for idx, s in tqdm(enumerate(data.sents), total=len(data.sents), ncols=70):
            raw_tokens = data.raw_tokens[idx]
            gold_spans = data.gold_spans[idx]
            gold_tags = data.gold_tags[idx]
            assert len(gold_spans) == len(gold_tags)
            if Baseline=='Random':
                pred_tree = random_parser(raw_tokens)
            elif Baseline=='Balanced':
                pred_tree=balance_parser(raw_tokens)
            elif Baseline=='Left_Branching':
                pred_tree=left_branching(raw_tokens)
            elif Baseline=='Right_Branching':
                pred_tree=right_branching(raw_tokens)



            ps = get_nonbinary_spans(get_predict_actions(pred_tree))[0]
            score.update(ps, gold_spans, gold_tags)
        score.derive_final_score()
        scores.append(score)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        default='../data/code_new/python_ast_new_new/valid_all.ast', type=str)
    parser.add_argument('--result-path', default='outputs', type=str)
    parser.add_argument('--from-scratch', default=False, action='store_true')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--bias', default=0, type=float,
                        help='the right-branching bias hyperparameter lambda')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--token-heuristic', default='mean', type=str,
                        help='Available options: mean, first, last')
    parser.add_argument('--use-not-coo-parser', default=False,
                        action='store_true',
                        help='Turning on this option will allow you to exploit '
                             'the NOT-COO parser (named by Dyer et al. 2019), '
                             'which has been broadly adopted by recent methods '
                             'for unsupervised parsing. As this parser utilizes'
                             ' the right-branching bias in its inner workings, '
                             'it may give rise to some unexpected gains or '
                             'latent issues for the resulting trees. For more '
                             'details, see https://arxiv.org/abs/1909.09428.')

    args = parser.parse_args()
    setattr(args, 'device', f'cuda:{args.gpu}'
    if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    setattr(args, 'time', datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'))

    dataset_name = args.data_path.split('/')[-1].split('.')[0]
    parser = '-w-not-coo-parser' if args.use_not_coo_parser else ''
    pretrained = 'scratch' if args.from_scratch else 'pretrained'
    result_path = f'{args.result_path}/{dataset_name}-{args.token_heuristic}'
    result_path += f'-{pretrained}-{args.bias}{parser}'
    now = datetime.datetime.now()
    date_suffix = '-'.join((str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond]))
    result_path+=f'-{date_suffix}'

    setattr(args, 'result_path', result_path)
    set_seed(args.seed)
    logging.disable(logging.WARNING)
    print('[List of arguments]')
    for a in args.__dict__:
        print(f'{a}: {args.__dict__[a]}')

    scores = evaluate(args)
    with open(f'{args.result_path}/scores.pickle', 'wb') as f:
        pickle.dump(scores, f)


if __name__ == '__main__':
    main()
