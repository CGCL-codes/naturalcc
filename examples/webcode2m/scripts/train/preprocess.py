import sys, os
sys.path.append(os.path.abspath('.'))
from utils import smart_tokenizer_and_embedding_resize
from datasets import load_dataset, load_from_disk
from scripts.train.my_dataset import UICoderDataset
from transformers import AutoProcessor, AddedToken, Pix2StructForConditionalGeneration, Pix2StructProcessor,Pix2StructImageProcessor
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, default="/data02/users/lz/code/UICoder/datasets/c4-wash/H128-2560_C128-4096_R2/c4-chunk05-format-scored-parquet") # 0-100
    parser.add_argument("--processor_path", "-p", type=str, default="/data02/models/pix2struct-large/") # 0-900
    parser.add_argument("--save_dir", "-o", type=str, default="./data/cc_tmp")
    parser.add_argument("--stage", "-s", type=int, default=0)  
    parser.add_argument("--max_length", "-ml", type=int, default=3072)
    parser.add_argument("--max_patches", "-mp", type=int, default=1024)
    parser.add_argument("--max_num", "-mn", type=int, default=100)
    args = parser.parse_args()

    dprocessor = Pix2StructProcessor.from_pretrained(args.processor_path)
    model = Pix2StructForConditionalGeneration.from_pretrained(args.processor_path,is_encoder_decoder=True)
    smart_tokenizer_and_embedding_resize(model, processor.tokenizer, {
        'bos_token': AddedToken('<s>', rstrip=False, lstrip=False, single_word=False, normalized=True),
    })
    ds = UICoderDataset(args.input_dir, preprocess=True,stage=0, processor=processor,max_length=args.max_length,max_patches=args.max_patches,max_num=args.max_num)
    ds.save(args.save_dir)
    