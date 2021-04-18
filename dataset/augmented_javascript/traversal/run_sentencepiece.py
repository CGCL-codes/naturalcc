import argparse
import sentencepiece as spm
import tqdm
import os
import gzip
import ujson
from dataset.augmented_javascript.utils.jsonl_dataset import JSONLinesDataset, normalize_docstring
from dataset.augmented_javascript.utils.util import normalize_program
from dataset.codesearchnet.utils.codebert_utils import vocab2dict

from dataset.augmented_javascript import (
    DATASET_DIR, RAW_DATA_DIR,
)


def make_corpus(input, output):
    dataset = JSONLinesDataset(input, {"function": "function", "docstring": "docstring"})
    print("Number of functions:", len(dataset))
    print("Example original:", dataset[0]["function"])
    print("Example normalized:", normalize_program(dataset[0]["function"]))
    print("Example normalized docstring:", normalize_docstring(dataset[0]["docstring"]))

    with open(output, "w", encoding='utf8') as f:
        for ex in tqdm.tqdm(dataset, "Writing corpus to txt"):
            # Write docstring
            if ex["docstring"]:
                print(normalize_docstring(ex["docstring"]), file=f)
            # Write normalized function
            function = ex["function"]
            line = normalize_program(function)
            print(line, file=f)

    print("Wrote corpus to:", output)


def spm_train(
    input: str, model_prefix: str, vocab_size: int, defined_tokens,
    character_coverage=0.9995, model_type='unigram'
):  # , input_sentence_size: int, shuffle_input_sentence: str):
    base_defined_symbols = '[CLS],[SEP],[MASK],[EOL],[URL]'.split(',')
    vocab_size += len(defined_tokens)
    user_defined_symbols = ','.join(base_defined_symbols + defined_tokens)
    command = f"--input={input} --model_prefix={model_prefix} --vocab_size={vocab_size} " \
              f"--character_coverage={character_coverage} --model_type={model_type} --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3" \
              f" --unk_piece=[UNK] --pad_piece=[PAD] --user_defined_symbols={user_defined_symbols} --hard_vocab_limit=false"
    print(command)
    spm.SentencePieceTrainer.Train(command)


if __name__ == "__main__":
    # fire.Fire({"make_corpus": make_corpus, "spm_train": spm_train})
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, default='piece', help='id(num)/piece(str)')
    parser.add_argument("--vocab-size", type=int, default=8000, help='token dictionary size')
    parser.add_argument("--src-file", type=str,
                        default=os.path.join(DATASET_DIR,
                                             'codebert/traverse_roberta/filter/javascript/train.ast.leaf_token'),
                        help='source data')
    parser.add_argument("--tgt-dir", type=str,
                        default=os.path.join(DATASET_DIR,
                                             'codebert/traverse_roberta/filter/javascript/data-mmap'),
                        help='save dir for sentencepiece bpe models or save files')
    parser.add_argument("--defined_tokens", type=str,
                        default=os.path.join(DATASET_DIR,
                                             'codebert/traverse_roberta/filter/javascript/.ast.node_types'),
                        help='save dir for sentencepiece bpe models or save files')
    parser.add_argument("--model-type", type=str, default='unigram', help='source data')
    parser.add_argument("--model-prefix", type=str, default='traversal', help='source data')
    parser.add_argument("--keep-empty", type=bool, default=True, help="keep empty lines")
    parser.add_argument("--overwrite", type=bool, default=False, help="build BPE model for files")
    parser.add_argument("--workers", type=int, default=100, help='multi-processors number')
    args = parser.parse_args()

    os.makedirs(args.tgt_dir, exist_ok=True)

    input = [os.path.expanduser(file) for file in args.src_file.split(',')]
    input = ','.join(input)
    args.defined_tokens = os.path.expanduser(args.defined_tokens)
    defined_tokens = []
    with open(args.defined_tokens, 'r', encoding='utf8') as reader:
        for line in reader:
            defined_tokens.append(ujson.loads(line)[0])

    # spm_train
    model_prefix = os.path.join(args.tgt_dir, args.model_prefix)
    spm_train(input, model_prefix=model_prefix, vocab_size=args.vocab_size, defined_tokens=defined_tokens,
              model_type=args.model_type)
