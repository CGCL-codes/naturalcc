import argparse
import os
from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
from torchmetrics.functional import accuracy

# Set the transformers logging to error only to suppress warnings
hf_logging.set_verbosity_error()


def memorization_probing(batch):
    secret_token_ids = []
    batch_track = []
    for i, ids in enumerate(batch['secret_token_ids']):
        input_ids = torch.LongTensor(ids)
        secret_token_ids.append(input_ids)
        batch_track.extend([i] * input_ids.shape[0])
    secret_token_ids = torch.cat(secret_token_ids, dim=0).to(device)
    batch_track = torch.LongTensor(batch_track)
    
    secret_prefix_token_ids = torch.cat([torch.LongTensor(ids) for ids in batch['secret_prefix_token_ids']], dim=0).to(device)

    preds = []
    for i in range(secret_token_ids.shape[-1]):
        # label = secret_token_ids[..., i]
        prompt = torch.cat([secret_prefix_token_ids, secret_token_ids[..., :i]], dim=-1)
        # pred = model.generate(prompt, max_length=secret_prefix_token_ids.shape[-1] + i + 1)[:, -1]
        pred = model.generate(prompt, max_new_tokens=1)[:, -1]
        preds.append(torch.squeeze(pred).detach().cpu())
        del prompt, pred
        torch.cuda.empty_cache()

    preds = torch.t(torch.stack(preds))
    secret_token_ids[secret_token_ids == tokenizer.pad_token_id] = -100
    ma = accuracy(preds, secret_token_ids.detach().cpu(), task='multiclass', num_classes=len(tokenizer.get_vocab()), ignore_index=-100, multidim_average='samplewise')

    batch['secret_MA_list'] = [None] * len(batch['secrets'])
    batch['secret_mean_MA'] = [None] * len(batch['secrets'])
    batch['secret_max_MA'] = [None] * len(batch['secrets'])
    for i in range(len(batch['secrets'])):
        batch['secret_MA_list'][i] = ma[batch_track == i]
        batch['secret_mean_MA'][i] = ma[batch_track == i].mean()
        batch['secret_max_MA'][i] = ma[batch_track == i].max()

    return batch


def main():
    dataset_path = f"./codeparrot-clean-train-secrets-probed-{args.model_name_or_path.split('/')[-1]}"
    if os.path.exists(dataset_path):
        ds_pii = load_from_disk(dataset_path)
        ds_pii_filter = ds_pii.filter(lambda example: example['secret_mean_MA'] >= 0.9, num_proc=48)
        print(ds_pii_filter)
        ds_pii_filter = ds_pii.filter(lambda example: example['secret_max_MA'] >= 0.9, num_proc=48)
        print(ds_pii_filter)
    else:
        if args.proxy:
            ds_pii = load_from_disk(f"./codeparrot-clean-train-secrets-probed-{args.proxy_model_name_or_path.split('/')[-1]}")
            ds_pii = ds_pii.filter(lambda example: example['secret_mean_MA'] >= 0.95, num_proc=16)
        else:
            ds_pii = load_from_disk(f"./codeparrot-clean-train-secrets-tokenized-{args.model_name_or_path.split('/')[-1]}")
        ds_pii = ds_pii.map(memorization_probing, batched=True, batch_size=args.batch_size)
        ds_pii.save_to_disk(dataset_path)
    print(ds_pii)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="codeparrot/codeparrot", type=str,
                        help="Model to train and evaluate, provide a repo name in Hugging Face hub or a local path.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use fp16 model precision.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which gpu to use if any')
    parser.add_argument("--batch_size", default=50, type=int,
                        help="Batch size.")
    parser.add_argument("--proxy", action='store_true',
                        help="Whether to use the proxy model.")
    parser.add_argument("--proxy_model_name_or_path", default="codeparrot/codeparrot-small", type=str,
                        help="Proxy model for memorization quantification.")
    args = parser.parse_args()
    print(args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        attention_dropout=0,
        # pad_token_id=tokenizer.pad_token_id,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = len(tokenizer)
    if hasattr(model, 'model'):
        print("Model has model.model.embed_tokens.padding_idx")
        model.model.embed_tokens.padding_idx = tokenizer.pad_token_id
    else:
        print("Model has get_input_embeddings.padding_idx")
        model.get_input_embeddings().padding_idx = tokenizer.pad_token_id
    if hasattr(model, 'lm_head'):
        print("Model has lm_head.padding_idx")
        model.lm_head.padding_idx = tokenizer.pad_token_id
        
    embedding_layer = model.get_input_embeddings()
    print(f"padding_idx: {embedding_layer.padding_idx}")

    assert model.config.pad_token_id == tokenizer.pad_token_id
    assert embedding_layer.num_embeddings == len(tokenizer)
    
    if args.fp16:
        model.half()
        
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if args.proxy:
        proxy_model = AutoModelForCausalLM.from_pretrained(
            args.proxy_model_name_or_path,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
            local_files_only=True
        )
        proxy_model.resize_token_embeddings(len(tokenizer))
        if args.fp16:
            proxy_model.half()
        proxy_model.to(device)

    main()
