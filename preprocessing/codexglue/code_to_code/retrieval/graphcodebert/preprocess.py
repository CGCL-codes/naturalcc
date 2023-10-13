# -*- coding: utf-8 -*-

import itertools
import os
from collections import Counter
from preprocessing.clcdsa.dfg import (
    extract_dataflow, parsers,
)
from preprocessing.codexglue.code_to_code.translation import (
    MODES,
)
from ncc import LOGGER
from ncc.data.dictionary import (
    TransformersDictionary,
)
from ncc.utils.file_ops import file_io
from ncc.utils.file_ops import json_io
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.path_manager import PathManager
from run.retrieval.graphcodebert.config.codetrans import config


def main(args):
    # task = tasks.get_task(args['preprocess']['task'])
    LOGGER.info('mkdir {} for {} task'.format(args['preprocess']['destdir'], args['preprocess']['task']))
    PathManager.mkdir(args['preprocess']['destdir'])
    vocab = TransformersDictionary.from_pretrained('microsoft/graphcodebert-base')

    # 2. ***************build dataset********************
    # dump into pkl file
    # transform a language's code into src format and tgt format simualtaneouly
    def parse_source_input(code, lang):
        code_tokens, dfg = extract_dataflow(code, parsers[lang], lang)
        code_tokens = vocab.subtokenize(code_tokens)

        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))

        # truncating
        code_tokens = code_tokens[
                      :config.MAX_TOKEN_LEN + config.MAX_DATA_FLOW_LEN - 3 - min(len(dfg), config.MAX_DATA_FLOW_LEN)] \
            [:512 - 3]
        source_tokens = [vocab.cls_token] + code_tokens + [vocab.sep_token]
        source_ids = vocab.convert_tokens_to_ids(source_tokens)
        position_idx = [i + vocab.pad() + 1 for i in range(len(source_tokens))]
        dfg = dfg[:config.MAX_TOKEN_LEN + config.MAX_DATA_FLOW_LEN - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for _ in dfg]
        source_ids += [vocab.unk() for _ in dfg]
        padding_length = config.MAX_TOKEN_LEN + config.MAX_DATA_FLOW_LEN - len(source_ids)
        position_idx += [vocab.pad()] * padding_length
        source_ids += [vocab.pad()] * padding_length

        # reindex
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([vocab.cls()])
        dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
        return [source_ids, position_idx, dfg_to_code, dfg_to_dfg]

    src_lang, tgt_lang = args['preprocess']['src_lang'], args['preprocess']['tgt_lang']
    for lang, mode in itertools.product([src_lang, tgt_lang], MODES):
        src_file = args['preprocess'][f'{mode}pref'].replace('*', lang) + ".code"
        dst_file = os.path.join(args['preprocess']['destdir'], lang, f"{mode}.pkl")
        PathManager.mkdir(os.path.dirname(dst_file))
        with file_io.open(src_file, 'r') as reader:
            keys = [
                'code', 'src_tokens', 'src_positions', 'dfg2code', 'dfg2dfg',
            ]
            data = {key: [] for key in keys}
            for line in reader:
                src_code = json_io.json_loads(line)
                # source_ids, source_mask
                src_line = parse_source_input(src_code, lang)
                for key, src in zip(keys, [src_code] + src_line):
                    data[key].append(src)

            # cp id
            data['proj_indices'] = [1] * len(data['code'])
            file_io.open(dst_file, mode='wb', data=data)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Downloading/Decompressing CodeSearchNet dataset(s) or Tree-Sitter Library(ies)")
    parser.add_argument(
        "--yaml_file", "-f", type=str, help="load {yaml_file}.yml for train",
        default='config/preprocess'
    )
    args = parser.parse_args()
    yaml_file = os.path.join(os.path.dirname(__file__), f"{args.yaml_file}.yml")
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    args = load_yaml(yaml_file)
    LOGGER.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
