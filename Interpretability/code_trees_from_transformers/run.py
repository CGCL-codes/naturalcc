import argparse
import datetime
import logging
import os
import pickle
from tqdm import tqdm

import torch

from transformers import RobertaTokenizer,RobertaModel,RobertaConfig
from code_dataset import Dataset
from code_measure import Measure
from code_parser import not_coo_parser,parser
from code_tools import set_seed,select_indices,group_indices
from code_yk import get_nonbinary_spans,get_predict_actions

# MODELS = [(BertModel, BertTokenizer, BertConfig, 'bert-base-cased'),
#           (BertModel, BertTokenizer, BertConfig, 'bert-large-cased'),
#           (GPT2Model, GPT2Tokenizer, GPT2Config, 'gpt2'),
#           (GPT2Model, GPT2Tokenizer, GPT2Config, 'gpt2-medium'),
#           (RobertaModel, RobertaTokenizer, RobertaConfig, 'roberta-base'),
#           (RobertaModel, RobertaTokenizer, RobertaConfig, 'roberta-large'),
#           (XLNetModel, XLNetTokenizer, XLNetConfig, 'xlnet-base-cased'),
#           (XLNetModel, XLNetTokenizer, XLNetConfig, 'xlnet-large-cased')]

MODELS=[(RobertaModel,RobertaTokenizer,RobertaConfig,'microsoft/codebert-base'),
        (RobertaModel,RobertaTokenizer,RobertaConfig,'microsoft/graphcodebert-base')
]

def evaluate(args):
    scores = dict()

    for model_class, tokenizer_class, model_config, pretrained_weights in MODELS:
        tokenizer = tokenizer_class.from_pretrained(
            pretrained_weights)
        if args.from_scratch:
            config = model_config.from_pretrained(pretrained_weights)
            config.output_hidden_states = True
            config.output_attentions = True
            model = model_class(config).to(args.device)
        else:
            model = model_class.from_pretrained(
                pretrained_weights,
                output_hidden_states=True,
                output_attentions=True).to(args.device)

        with torch.no_grad():
            test_sent = tokenizer.encode('test', add_special_tokens=False)
            token_ids = torch.tensor([test_sent]).to(args.device)
            all_hidden, all_att = model(token_ids)[-2:]
            n_layers = len(all_att)
            n_att = all_att[0].size(1)
            n_hidden = all_hidden[0].size(-1)

        measure = Measure(n_layers, n_att)  #initial score
        data = Dataset(path=args.data_path, tokenizer=tokenizer)

        for idx, s in tqdm(enumerate(data.sents), total=len(data.sents),
                           desc=pretrained_weights, ncols=70):
            raw_tokens = data.raw_tokens[idx]
            raw_tokens_text=data.raw_tokens_text[idx]
            raw_tokens_text_number=data.raw_tokens_text_number[idx]
            tokens = data.tokens[idx] 
            if len(raw_tokens) < 2:
                data.cnt -= 1
                continue
            token_ids = tokenizer.encode(s, add_special_tokens=False)
            token_ids_tensor = torch.tensor([token_ids]).to(args.device)
            with torch.no_grad():
                all_hidden, all_att = model(token_ids_tensor)[-2:]
            all_hidden, all_att = list(all_hidden[1:]), list(all_att)

            # (n_layers, seq_len, hidden_dim)
            all_hidden = torch.cat([all_hidden[n] for n in range(n_layers)], dim=0)
            # (n_layers, n_att, seq_len, seq_len)
            all_att = torch.cat([all_att[n] for n in range(n_layers)], dim=0)

            if len(tokens) > len(raw_tokens):
                th = args.token_heuristic
                if th == 'first' or th == 'last':
                    mask = select_indices(tokens, raw_tokens, pretrained_weights, th)
                    assert len(mask) == len(raw_tokens)
                    all_hidden = all_hidden[:, mask]
                    all_att = all_att[:, :, mask, :]
                    all_att = all_att[:, :, :, mask]
                else:
                    mask = group_indices(tokens, raw_tokens, pretrained_weights)
                    raw_seq_len = len(raw_tokens)
                    all_hidden = torch.stack(
                        [all_hidden[:, mask == i].mean(dim=1)
                         for i in range(raw_seq_len)], dim=1)
                    all_att = torch.stack(
                        [all_att[:, :, :, mask == i].sum(dim=3)
                         for i in range(raw_seq_len)], dim=3)
                    all_att = torch.stack(
                        [all_att[:, :, mask == i].mean(dim=2)
                         for i in range(raw_seq_len)], dim=2)

            all_hidden=torch.stack([all_hidden[:,i] for i in raw_tokens_text_number],dim=1)
            all_att=torch.stack([all_att[:,:,i] for i in raw_tokens_text_number],dim=2)


            l_hidden, r_hidden = all_hidden[:, :-1], all_hidden[:, 1:]
            l_att, r_att = all_att[:, :, :-1], all_att[:, :, 1:]
            syn_dists = measure.derive_dists(l_hidden, r_hidden, l_att, r_att)
            gold_spans = data.gold_spans[idx]
            gold_tags = data.gold_tags[idx]
            assert len(gold_spans) == len(gold_tags)

            for m, d in syn_dists.items():
                pred_spans = []
                for i in range(measure.scores[m].n):
                    dist = syn_dists[m][i].tolist()

                    if len(dist) > 1:
                        bias_base = (sum(dist) / len(dist)) * args.bias
                        bias = [bias_base * (1 - (1 / (len(dist) - 1)) * x)
                                for x in range(len(dist))]
                        dist = [dist[i] + bias[i] for i in range(len(dist))]

                    if args.use_not_coo_parser:
                        pred_tree = not_coo_parser(dist, raw_tokens_text)
                    else:
                        pred_tree = parser(dist, raw_tokens_text)
                        # if 'if' in raw_tokens and len(dist)<20 and m=='avg_jsd' and i==7:
                        #     print(pred_tree)
                        #     print(dist)
                        #     print(raw_tokens)

                    ps = get_nonbinary_spans(get_predict_actions(pred_tree))[0]
                    pred_spans.append(ps)

                measure.scores[m].update(pred_spans, gold_spans, gold_tags)

        measure.derive_final_score()
        scores[pretrained_weights] = measure.scores

        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)

        with open(f'{args.result_path}/{pretrained_weights.split("/")[-1]}.txt', 'w') as f:
            print('Model name:', pretrained_weights, file=f)
            print('Experiment time:', args.time, file=f)
            print('# of layers:', n_layers, file=f)
            print('# of attentions:', n_att, file=f)
            print('# of hidden dimensions:', n_hidden, file=f)
            print('# of processed sents:', data.cnt, file=f)
            max_corpus_f1, max_sent_f1 = 0, 0
            for n in range(n_layers):
                print(f'[Layer {n + 1}]', file=f)
                print('-' * (119 + measure.max_m_len), file=f)
                for m, s in measure.scores.items():
                    if m in measure.h_measures + measure.a_avg_measures:
                        print(
                            f'| {m.upper()} {" " * (measure.max_m_len - len(m))} '
                            f'| Corpus F1: {s.corpus_f1[n] * 100:.2f} '
                            f'| Sent F1: {s.sent_f1[n] * 100:.2f} ',
                            end='', file=f)
                        for z in range(len(s.label_recalls[0])):
                            print(
                                f'| {s.labels[z]}: '
                                f'{s.label_recalls[n][z] * 100:.2f} ',
                                end='', file=f)
                        print('|', file=f)
                        if s.sent_f1[n] > max_sent_f1:
                            max_corpus_f1 = s.corpus_f1[n]
                            max_sent_f1 = s.sent_f1[n]
                            max_measure = m
                            max_layer = n + 1
                    else:
                        for i in range(n_att):
                            m_att = str(i) if i > 9 else '0' + str(i)
                            m_att = m + m_att + " " * (
                                    measure.max_m_len - len(m))
                            i_att = n_att * n + i
                            print(
                                f'| {m_att.upper()}'
                                f'| Corpus F1: {s.corpus_f1[i_att] * 100:.2f} '
                                f'| Sent F1: {s.sent_f1[i_att] * 100:.2f} ',
                                end='', file=f)
                            for z in range(len(s.label_recalls[0])):
                                print(f'| {s.labels[z]}: '
                                      f'{s.label_recalls[i_att][z] * 100:.2f} ',
                                      end='', file=f)
                            print('|', file=f)
                            if s.sent_f1[i_att] > max_sent_f1:
                                max_corpus_f1 = s.corpus_f1[i_att]
                                max_sent_f1 = s.sent_f1[i_att]
                                max_measure = m_att
                                max_layer = n + 1
                    print('-' * (119 + measure.max_m_len), file=f)
            print(f'[MAX]: | Layer: {max_layer} '
                  f'| {max_measure.upper()} '
                  f'| Corpus F1: {max_corpus_f1 * 100:.2f} '
                  f'| Sent F1: {max_sent_f1 * 100:.2f} |')
            print(f'[MAX]: | Layer: {max_layer} '
                  f'| {max_measure.upper()} '
                  f'| Corpus F1: {max_corpus_f1 * 100:.2f} '
                  f'| Sent F1: {max_sent_f1 * 100:.2f} |', file=f)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',
                        default='../data/code_java/java_ast_new_new/test.ast', type=str)
    parser.add_argument('--result-path', default='outputs', type=str)
    parser.add_argument('--from-scratch', default=False, action='store_true')
    parser.add_argument('--gpu', default=3, type=int)
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