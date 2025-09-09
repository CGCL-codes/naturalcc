import argparse
import feather
import pandas as pd
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smoothie = SmoothingFunction().method4


# calculate BLEU-4 score
def calc_bleu4(tokenizer, sample, generated):
    ref = tokenizer.decode(sample)
    hyp = tokenizer.decode(generated)
    return sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)


def memorization_extraction(args):
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    
    base_tokenizer = AutoTokenizer.from_pretrained(
            'Salesforce/codegen-350M-multi',
            padding_side='left',
            # add_special_tokens=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        attention_dropout=0,
        # pad_token_id=tokenizer.eos_token_id,
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
    model.to(device)

    df = feather.read_dataframe('benchmark.feather')
    if base_tokenizer.vocab != tokenizer.vocab:
        print('Different tokenizers: Re-encoding samples...')
        df['sample'] = df['sample'].apply(lambda x: tokenizer.encode(base_tokenizer.decode(x, skip_special_tokens=True)))
        df = df[df['sample'].apply(len) >= 100].reset_index(drop=True) # drop samples that are too short
    else: # same tokenizer
        print('Same tokenizers: No need to re-encode samples...')
    df['prefix'] = df['sample'].apply(lambda x: x[:64])
    df['suffix'] = df['sample'].apply(lambda x: x[64:128])

    gen_suffix = []
    # iterate with batch size
    with torch.no_grad():
        for i in tqdm(range(0, len(df), args.batch_size)):
            batch = torch.tensor(df.iloc[i: i + args.batch_size].prefix.tolist()).to(device)
            # output = model.generate(batch, max_length=128)[..., 64:].tolist()
            output = model.generate(batch, max_new_tokens=64)[..., 64:].tolist()
            gen_suffix.extend(output)

    df['gen_suffix'] = gen_suffix
    df['bleu4'] = df.apply(lambda x: calc_bleu4(tokenizer, x['suffix'], x['gen_suffix']), axis=1)
    
    memorization_df = df[df['bleu4'] >= 0.95]
    memorization_df.rename(columns={'index': 'doc_id'}, inplace=True)
    memorization_df['text'] = memorization_df['sample'].apply(lambda x: tokenizer.decode(x, skip_special_tokens=True))
    memorization_df['corpus'] = 'BigQuery'
    memorization_df = memorization_df[['doc_id', 'hash', 'copies', 'corpus', 'text', 'bleu4']]
    print(memorization_df)
    model_name = args.model_name_or_path.split('/')[-1]
    memorization_df.to_csv(f'{model_name}_memorization.csv', index=False, encoding='utf-8')

    random.seed(42)
    memorization_df_indexes = list(range(len(memorization_df)))
    random.shuffle(memorization_df_indexes)
    sampled_df = memorization_df.iloc[memorization_df_indexes[:args.k], :]
    sampled_df.to_csv(f'../unlearning/data/{model_name}_secret/{model_name}_retained_set_{args.k}.csv', index=False, encoding='utf-8')


def main():
    # Parsing Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Salesforce/codegen-350M-mono", type=str,
                        help="Model to train and evaluate, provide a repo name in Hugging Face hub or a local path.")
    parser.add_argument('--gpu_id', type=str, default="0", help="specify the GPU id")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use fp16 model precision.")
    parser.add_argument('--k', type=int, default=32,
                        help="The number of forgotten samples.")
    args = parser.parse_args()
    
    memorization_extraction(args)


if __name__ == '__main__':
    main()
