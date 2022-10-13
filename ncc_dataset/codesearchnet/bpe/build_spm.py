import argparse
import os

import sentencepiece as spm

from ncc_dataset.codesearchnet import DEDUPE_DIR
from ncc import LOGGER
from ncc.data import constants


def spm_train(input: str, model_prefix: str, vocab_size: int, character_coverage=1.0, model_type='unigram',
              special_symbols=None):
    special_symbols = ','.join(special_symbols)
    command = f"--input={input} --model_prefix={model_prefix} --vocab_size={vocab_size} " \
              f"--character_coverage={character_coverage} --model_type={model_type} " \
              f"--pad_piece={constants.PAD} --pad_id=0 " \
              f"--bos_piece={constants.BOS} --bos_id=1 " \
              f"--eos_piece={constants.EOS} --eos_id=2 " \
              f"--unk_piece={constants.UNK} --unk_id=3 " \
              f"--user_defined_symbols={special_symbols} --hard_vocab_limit=false --train_extremely_large_corpus=true"
    LOGGER.info(command)
    spm.SentencePieceTrainer.Train(command)


if __name__ == "__main__":
    # python -m dataset.csn.codebert.run_sentencepiece --src-dir ~/CodeSearchNet/flatten --tgt-dir ~/CodeSearchNet/codebert/ --vocab-size 50000 --model-type bpe --model-prefix codesearchnet
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, default='piece', help='id(num)/piece(str)')
    parser.add_argument("--vocab-size", type=int, default=50000, help='token dictionary size')
    parser.add_argument("--src-dir", type=str, default=DEDUPE_DIR, help='source data')
    parser.add_argument("--language", type=str, help='sentencepiece tokenizer for language')
    # parser.add_argument("--corpus_modalities", type=list, help='sentencepiece tokenizer for modalities')
    parser.add_argument("--tgt-dir", type=str, default=DEDUPE_DIR,
                        help='save dir for sentencepiece bpe models or save files')
    # parser.add_argument("--bpe-dir", type=str, default='wordpiece_bpe', help='wordpiece_bpe modal save direction')
    parser.add_argument("--model-type", type=str, default='unigram', help='source data')
    parser.add_argument("--model-prefix", type=str, default='csn', help='source data')
    parser.add_argument("--keep-empty", type=bool, default=True, help="keep empty lines")
    parser.add_argument("--overwrite", type=bool, default=False, help="build BPE model for files")
    parser.add_argument("--workers", type=int, default=999, help='multi-processors number')
    args = parser.parse_args()

    corpus_file = os.path.join(DEDUPE_DIR, 'data.txt')
    special_symbols = ['[CLS]', '[SEP]', '[MASK]', '[EOL]', '[URL]']
    model_prefix = os.path.join(args.tgt_dir, f'{args.model_prefix}.spm')
    spm_train(corpus_file, model_prefix=model_prefix, vocab_size=args.vocab_size, model_type=args.model_type,
              special_symbols=special_symbols)
