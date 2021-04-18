import argparse
import sentencepiece as spm
import tqdm
import os
from dataset.augmented_javascript.utils.jsonl_dataset import JSONLinesDataset, normalize_docstring
from dataset.augmented_javascript.utils.util import normalize_program
from dataset.codesearchnet.utils.codebert_utils import vocab2dict


def explore(input):
    dataset = JSONLinesDataset(input, {"function": "function", "docstring": "docstring"})
    print("Number of functions:", len(dataset))
    print("Example original:", dataset[0]["function"])
    print("Example normalized:", normalize_program(dataset[0]["function"]))
    print("Example normalized docstring:", normalize_docstring(dataset[0]["docstring"]))
    count_w_docstring, count_wo_docstring = 0, 0
    for ex in tqdm.tqdm(dataset, "Writing corpus to txt"):
        # Write docstring
        if ex["docstring"]:
            # f.write(normalize_docstring(ex["docstring"]) + "\n")
            count_w_docstring += 1
        # # Write normalized function
        # function = ex["function"]
        # line = normalize_program(function)
        # f.write(line + "\n")
    print('count_w_docstring: ', count_w_docstring)


if __name__ == "__main__":
    # fire.Fire({"make_corpus": make_corpus, "spm_train": spm_train})
    parser = argparse.ArgumentParser()
    # parser.add_argument("--format", type=str, default='piece', help='id(num)/piece(str)')
    # parser.add_argument("--vocab-size", type=int, default=8000, help='token dictionary size')
    parser.add_argument("--src-dir", type=str, default='~/.ncc/augmented_javascript/raw', help='source data')
    # parser.add_argument("--tgt-dir", type=str, default='~/.ncc/augmented_javascript/contracode/data-raw', help='save dir for sentencepiece bpe models or save files')
    # parser.add_argument("--model-type", type=str, default='unigram',  help='source data')
    # parser.add_argument("--model-prefix", type=str, default='csnjs_8k_9995p_unigram_url',  help='source data')
    #
    # # parser.add_argument("--bpe-dir", type=str, default='wordpiece_bpe', help='wordpiece_bpe modal save direction')
    # parser.add_argument("--keep-empty", type=bool, default=True, help="keep empty lines")
    # parser.add_argument("--overwrite", type=bool, default=False, help="build BPE model for files")
    # # parser.add_argument("--insert", type=bool, help='insert CLS/S_SEP')
    # parser.add_argument("--workers", type=int, default=16, help='multi-processors number')
    args = parser.parse_args()

    args.src_dir = os.path.expanduser(args.src_dir)
    # args.tgt_dir = os.path.expanduser(args.tgt_dir)

    input = os.path.join(args.src_dir, 'javascript_dedupe_definitions_nonoverlap_v2_train.jsonl')
    # output = os.path.join(args.src_dir, 'javascript_dedupe_definitions_nonoverlap_v2_train.txt')

    explore(input)
