import argparse
import random
import logging
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F
from dataset import EvalDataset

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer)
from model import RNNModel
        
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'rnn': (GPT2Config, RNNModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
}

def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
     
        
def rank_list(list, descending=True):
    """Return the rank(torch tensor) of a list

    Args:
        list: Torch tensor: [vocabulary_size], prediction scores of the model (scores for each vocabulary token).
        descending: descending or ascending. Defaults to descending
    """
    ranks = torch.empty_like(list, dtype=torch.int64)
    ranks[torch.argsort(list, descending=descending)] = torch.arange(len(list))
    return ranks


def greater_than(list, value):
    """Return how many value greater than `value` in the input list

    Args:
        list (torch.Tensor): 
        value (int or torch.Tensor(size == [1])): 
    """
    return torch.sum(list.gt(value)).item()

def post_process(args, gts, preds, ranks, probability, label, tokenizer, saved_file):
    """Split the prediction by file level, filter the special tokens(<s>, </s>, <pad>, <EOL>...), save the results in a .pth file

    Args:
        args:
        gts: ground truth tokens' index
        preds: prediction tokens' index     
        ranks: ground truth tokens' predicted ranks
        probability: ground truth tokens' predicted probability
        label: 1: member, 0: non-member
        tokenizer: get the special tokens' index: tokenzier.bos_token_id
        saved_file: the saved pth file.
    """
    cnt = 0
    results = []
    
    cur_gt, cur_pred, cur_rank, cur_probability = [], [], [], []
    for i, (gt, pred, rank, prob) in enumerate(zip(gts, preds, ranks, probability)):
        if gt in [tokenizer.bos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if gt == tokenizer.eos_token_id:
            cnt += 1
            results.append({"gt": torch.tensor(cur_gt), "pred": torch.tensor(cur_pred),
                            "rank": torch.tensor(cur_rank), "prob": torch.tensor(cur_probability),
                            "label": torch.tensor(label)})
            cur_gt, cur_pred, cur_rank, cur_probability = [], [], [], []
        else:
            cur_gt.append(gt)
            cur_pred.append(pred)
            cur_rank.append(rank)
            cur_probability.append(prob)
            
    torch.save(results, saved_file)
    return cnt
                  
def load_rank(args, model, tokenizer, member="train", non_member="test"):
    """Load samples' predicted ranks given a model, tokens will be not merged(considering the BPE)

    Args:
        args:
        model: code completion model(GPT-2)
        tokenizer: tokenzieation
        member: member dataset. Defaults to "train".
        non_member: non-member dataset. Defaults to "test".
    """   
    model.to(args.device)
    model.eval()
    # member samples' label is 1, non-member samples' label is 0
    for file_type, label in zip((member, non_member), (1, 0)):
        logger.info("load the predicted ranks of the {} dataset({})".format("member" if label else "non-member",
                                                                            os.path.join(args.data_dir, f"{file_type}.txt")))
        eval_dataset = EvalDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        total_pred = []
        total_gt = []
        total_ranks = [] # the predicted ranks of ground truth

        total_probability = [] # the predicted probability of the ground truth
        
        prev_pred = None
        prev_pred_score = None
        prev_pred_prob = None
        # prev_rank_list = None

        for step, batch in enumerate(eval_dataloader):
            inputs = batch.to(args.device)
            
            with torch.no_grad():
                outputs = model(inputs)
                pred_scores = outputs[0]
                pred_probs = F.softmax(pred_scores, dim=-1)
                pred_ids = pred_scores.argmax(-1)
            
            for pred, pred_score, pred_prob, gt in zip(pred_ids, pred_scores, pred_probs, inputs):
                pred = pred.cpu().tolist()
                gt = gt.cpu().tolist()
                pred_score = pred_score.cpu()
                pred_prob = pred_prob.cpu()
                
                for i, y in enumerate(gt):
                    total_gt.append(y)
                    if i == 0:
                        if y == tokenizer.bos_token_id:
                            total_pred.append(tokenizer.bos_token_id)
                            total_ranks.append(-1) # when <s> is the first character, it will not be predicted
                            total_probability.append(0)
                        else:
                            total_pred.append(prev_pred) # when the first character is not <s>, the first character is predicted by the previous sample's the last prediction        
                            total_ranks.append(greater_than(prev_pred_score, prev_pred_score[y]))
                            total_probability.append(prev_pred_prob[y].item())
                            # total_probability.append(F.softmax(prev_pred_score, dim=-1)[y].item())
                            # total_ranks.append(prev_rank_list[y].item())
                    else:
                        total_pred.append(pred[i-1])
                        total_ranks.append(greater_than(pred_score[i-1], pred_score[i-1][y]))
                        total_probability.append(pred_prob[i-1][y].item())
                        # total_probability.append(F.softmax(pred_score[i-1], dim=-1)[y].item())
                        # total_ranks.append(rank_list(pred_score[i-1])[y].item()) # get the ground truth's predicted rank
                    
                    if i == len(gt)-1:
                        prev_pred = pred[i]
                        prev_pred_score = pred_score[i]
                        prev_pred_prob = pred_prob[i]
                        # prev_rank_list = rank_list(pred_score[i])
            
            if step % args.logging_steps == 0:
                logger.info(f"{step} are done!")
        
        assert len(total_gt) == len(total_pred) and len(total_pred) == len(total_ranks) and len(total_ranks) == len(total_probability)
        
        saved_file = os.path.join(args.output_dir, "{}_ranks.pth".format("member" if label == 1 else "non_member"))
        total_samples = post_process(args, total_gt, total_pred, total_ranks, total_probability, label, tokenizer, saved_file)
        logger.info("Eval on {}, saved at {}".format(total_samples, saved_file))
            
                    
def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--langs", default=None, type=str, required=True,
                        help="Languages to train, if all, train all languages in data_dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--lit_file", type=str,
                        help="literals json file")


    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X evaluation steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    parser.add_argument('--log_file', type=str, default='')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(filename)s - %(funcName)s - %(lineno)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Use FileHandler to output log to a file
    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -  %(filename)s - %(funcName)s - %(lineno)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
    logger.addHandler(fh)

    logger.warning("device: %s, n_gpu: %s",args.device, args.n_gpu)
    # Set seed
    set_seed(args)
    
    # get special tokens
    special_tokens = get_special_tokens(args.lit_file)
    
    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained = args.pretrain_dir
    if pretrained:
        tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
        if args.model_type == "rnn":
            model = model_class(len(tokenizer), 768, 768, 1)
            model_last = os.path.join(pretrained, 'model.pt')
            if os.path.exists(model_last):
                logger.warning(f"Loading model from {model_last}")
                model.load_state_dict(torch.load(model_last, map_location="cpu")) 
        else:
            model = model_class.from_pretrained(pretrained)
            model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_dir, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
        args.vocab_size = len(tokenizer)
        if args.model_type == "rnn":
            model = model_class(len(tokenizer), 768, 768, 1)
        else:
            config = config_class.from_pretrained(args.config_dir)
            model = model_class(config)
            model.resize_token_embeddings(len(tokenizer))

    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")
    
    logger.info("Loading ranks parameters %s", args)
    
    load_rank(args, model, tokenizer, member="train", non_member="test")
    


if __name__ == "__main__":
    main()