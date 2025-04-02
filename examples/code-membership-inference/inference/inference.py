import argparse
import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.optim import Adam
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

from dataset import AuditDataset
from model import AuditModel

logger = logging.getLogger(__name__)


def rank_to_histogram(ranks, bins, output_size):
    """convert the ranks list to histogram

    Args:
        ranks: rank list
        bins (int): the bins of the returned histogram
        output_size (int): the range of the returned histogram is [0, output_size]
    """
    histogram_bins = bins - 1 # an additional feature that counts how many times the ground-truth words are not predicted among the model's output size
    histogram = torch.histc(ranks, histogram_bins, min=0, max=output_size).cpu().tolist()
    histogram.append(len(ranks) - sum(histogram))
    return torch.tensor(histogram) / len(ranks)

def prob_to_histogram(probs, ranks, bins, output_size):
    """convert the probability list to histogram
    
     Args:
        probs: probability list
        ranks: rank list
        bins (int): the bins of the returned histogram
        output_size (int): the token whose rank exceeds the output size will be ignored
    """
    histogram_bins = bins - 1 # an additional feature that counts how many times the ground-truth words are not predicted among the model's output size
    
    # new_probs = torch.empty_like(probs)
    # for i in range(len(new_probs)):
    #     if ranks[i] > output_size:
    #         new_probs[i] = -1
    #     else:
    #         new_probs[i] = probs[i] * 100
    mask = ranks.le(output_size)
    new_probs = torch.masked_select(probs*100, mask) # filter tokens whose rank exceeds the output size
    histogram = torch.histc(new_probs, histogram_bins, min=0, max=100).cpu().tolist()
    histogram.append(len(probs) - sum(histogram))
    return torch.tensor(histogram) / len(probs)


def sample_queries(ranks, gts, vocab, number, sample_strategy="random"):
    """query the completion model with limited number of times, with different sample strategies

    Args:
        ranks: rank list
        gts: ground truth token list
        vocab: vocabulary frequency
        number: number of queries
        sample_strategy: sample strategy is random or not, ["random", "frequency"]
    """
    # don't do sampling
    if number <= 0 or number >= len(ranks):
        return ranks
    
    else:
        mask = torch.tensor([False for i in range(len(ranks))])
        if sample_strategy == "random":
            mask[:number] = True

            idx = torch.randperm(len(ranks))
            mask = mask[idx]
            return torch.masked_select(ranks, mask)
        
        elif sample_strategy == "frequency":            
            token_frequency = []
            for index, token in enumerate(gts):
                if token.item() in vocab:
                    token_frequency.append([index, vocab[token.item()]])
            
            token_frequency = sorted(token_frequency, key=lambda k : k[1], reverse=True)

            for i in range(number):
                if i < len(token_frequency):
                    mask[token_frequency[i][0]] = True

            return torch.masked_select(ranks, mask)

        
        else:
             raise NotImplementedError(f"the {sample_strategy} strategy has not been implemented")



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def train(args, train_dataset, model, test_dataset = None):
    """train a audit model"""
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    optimizer = Adam(model.parameters(), lr=0.01)
    num_examples = len(train_dataset)
    
    # Train
    model.to(args.device)
    set_seed(args)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Num epoch = %d", args.epoch)
    logger.info("  Batch size = %d", args.batch_size)
    for i in range(args.epoch):
        logger.info("  Epoch %d", i)
        for batch_idx, batch in enumerate(train_dataloader):
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            
            loss = outputs[1]
            
            # optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % args.logging_steps == 0:
                loss, current = loss.item(), batch_idx * args.batch_size
                logger.info(f"loss: {loss:>7f}  [{current:>5d}/{num_examples:>5d}]")
        if test_dataset:
            accuracy = evaluate(args, test_dataset, model)
            logger.info(f"test accuracy: {accuracy[0]}")
    # save the model to disk
    
    model_name = f"{args.number_of_shadow_model}-{args.shadow_model_type}-{args.output_size}{'-probability' if args.use_probability else ''}.pth"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))
    logger.info(f"save the model in {os.path.join(args.output_dir, model_name)}")
        
def evaluate(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    
    logger.info("***** Running evaluation {} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    model.to(args.device)
    set_seed(args)
    model.eval()
    
    ans = []
    gt = []
    pred_prob = []
    for batch in eval_dataloader:
        inputs, labels = batch[0], batch[1]
        inputs = inputs.to(args.device)
        gt.extend(labels.tolist())
                
        with torch.no_grad():
            ret = model.predict(inputs)
            ans.extend(ret[0])
            pred_prob.extend(ret[1][:,1].tolist())

    return accuracy_score(gt, ans), roc_auc_score(gt, pred_prob), precision_score(gt, ans), recall_score(gt, ans)
        

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="", type=str, 
                        help="The downloaded data path")
    parser.add_argument("--output_dir", default="", type=str, 
                        help="The output directory")
    parser.add_argument("--bins", default=100, type=int,
                        help="The bins of the rank list's histogram ")
    parser.add_argument("--hidden_states", default=50, type=int,
                        help="The hidden states of the MLP")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--logging_steps', type=int, default=400,
                    help="Log every X updates steps.")
    
    parser.add_argument("--do_train", default=True, action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action='store_true',
                        help="Whether to run eval on the dev set.")
        
    parser.add_argument("-nshdow", "--number_of_shadow_model", default=10, type=int,
                        help="The number of shadow model")
    parser.add_argument("--shadow_model_type", default="lstm", type=str,
                        help="The architecture of shadow models.", choices=["lstm", "code-gpt"])
    parser.add_argument("--target_model_type", default="lstm", type=str,
                        help="The architecture of target model.", choices=["lstm", "code-gpt"])
    parser.add_argument("--use_probability", default=False, action="store_true",
                        help="Whether use probability to audit or not")
    # parser.add_argument("--cross_model", default=True, action="store_true",
    #                     help="whether the target model type is the same as the shadow model type")
    
    parser.add_argument("--output_size", default=100002, type=int,
                        help="The output size of the model") # code-gpt's vocabulary's size is 50234, lstm's vocabulary's size is 100002
    
    parser.add_argument("--number_of_queries", default=0, type=int)
    parser.add_argument("--sampling_strategy", default="frequency", choices=["random", "frequency"])
    
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(filename)s - %(funcName)s - %(lineno)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("Training/evaluation parameters %s", args)

    args.cross_model = (args.shadow_model_type != args.target_model_type)
    set_seed(args)
    
    if args.do_train:
        # load the ranks dataset of the shadow model
        train_ranks, train_probability, train_labels = [], [], []
        test_ranks, test_probability, test_labels = [],[], []
        train_ratio, test_ratio = 0.8, 0.2
        for i in range(args.number_of_shadow_model):
            logger.info(f"Load the rank dataset of the shadow_{i} model")
            data_dir = os.path.join(args.data_dir, f"shadow_{i}/{args.shadow_model_type}/ranks")
            for label in ("member", "non_member"):
                data = torch.load(os.path.join(data_dir, f"{label}_ranks.pth"))
                data_size = len(data)
                
                train_ranks.extend([sample['rank'] + 1 for sample in data[: int(data_size*train_ratio)]])
                train_labels.extend([sample['label'] for sample in data[: int(data_size*train_ratio)]])
                test_ranks.extend([sample['rank'] + 1 for sample in data[int(data_size*train_ratio): ]])
                test_labels.extend([sample['label'] for sample in data[int(data_size*train_ratio): ]])

                if args.use_probability:
                    train_probability.extend([sample['prob'] for sample in data[: int(data_size*train_ratio)]])
                    test_probability.extend([sample['prob'] for sample in data[int(data_size*train_ratio): ]])
        if args.use_probability:
            train_dataset = AuditDataset([prob_to_histogram(train_prob.to(args.device), train_rank.to(args.device), args.bins, args.output_size) for train_prob, train_rank in zip(train_probability, train_ranks)], train_labels)
            test_dataset = AuditDataset([prob_to_histogram(test_prob.to(args.device), test_rank.to(args.device), args.bins, args.output_size) for test_prob, test_rank in zip(test_probability, test_ranks)], test_labels)
        else:
            train_dataset =  AuditDataset([rank_to_histogram(train_rank.to(args.device), args.bins, args.output_size) for train_rank in train_ranks], train_labels)
            test_dataset = AuditDataset([rank_to_histogram(test_rank.to(args.device), args.bins, args.output_size) for test_rank in test_ranks], test_labels)
        audit_model = AuditModel(args.bins, args.hidden_states)
        
        train(args, train_dataset, audit_model, test_dataset=test_dataset)
    
    if args.do_eval:
        audit_model = AuditModel(args.bins, args.hidden_states)
        if not args.cross_model:
            model_name = f"{args.number_of_shadow_model}-{args.shadow_model_type}-{args.output_size}{'-probability' if args.use_probability else ''}.pth"
        else:
            model_name =  f"{args.number_of_shadow_model}-{args.shadow_model_type}{'-probability' if args.use_probability else ''}.pth"
        model_path = os.path.join(args.output_dir, model_name)
        if not os.path.exists(model_path):
            logger.warning(f"don't find model in {model_path}")        
        else:
            logger.info(f"load the model from {model_path}")
            audit_model.load_state_dict(torch.load(model_path))
            
            # load the ranks dataset of the target model
            test_ranks, test_probability, test_labels = [], [], [] 
            logger.info(f"load the rank dataset of the target model")          
            data_dir = os.path.join(args.data_dir, f"target/{args.target_model_type}/ranks")
            
            for label in ("member", "non_member"):
                data = torch.load(os.path.join(data_dir, f"{label}_ranks.pth"))

                # load the word frequency
                with open(os.path.join(args.data_dir, f"target/{args.target_model_type}/{label}_word_frequency.txt")) as f:
                    word_frequency = json.loads(f.read().strip())
                
                token_list = dict() # key is the token's index, value is the token's frequency rank
                for rank, token in enumerate(word_frequency):
                    token_list[token[0]] = rank

                test_ranks.extend([sample_queries(sample['rank'] + 1, sample['gt'], token_list, args.number_of_queries, args.sampling_strategy) for sample in data]) 
                test_labels.extend([sample['label'] for sample in data])

                if args.use_probability:
                    test_probability.extend([sample['prob'] for sample in data]) # todo: sample the probability 
            
            if args.use_probability:
                test_dataset = AuditDataset([prob_to_histogram(test_prob.to(args.device), test_rank.to(args.device), args.bins, args.output_size) for test_prob, test_rank in zip(test_probability, test_ranks)], test_labels)
            else:
                test_dataset = AuditDataset([rank_to_histogram(test_rank.to(args.device), args.bins, args.output_size) for test_rank in test_ranks], test_labels)
            accuracy, auc, precision, recall = evaluate(args, test_dataset, audit_model)
            logger.info("*****  evaluation finished  *****")
            logger.info(f"accuracy: {accuracy}, auc: {auc}, precision: {precision}, recall: {recall}")
            logger.info(f"accuracy: {round(accuracy, 4)*100}%, auc: {round(auc, 4)*100}%, precision: {round(precision, 4)*100}%, recall: {round(recall, 4)*100}%")      

        
if __name__ == "__main__":
    main()