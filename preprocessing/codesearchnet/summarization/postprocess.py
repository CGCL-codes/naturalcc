import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser("post process for codesearchnet")
    parser.add_argument('-d', '--dataset-dir', default='data-bin/codesearchnet', 
                        help='directory where your binary dataset locates')
    parser.add_argument('-s', '--source-lang', default='code_tokens', help='source language')    
    parser.add_argument('-t', '--target-lang', default='docstring_tokens', help='target language')

    args = parser.parse_args()
    
    dict_file = 'dict.{}.txt'.format(args.source_lang)
    dict_file = os.path.join(args.dataset_dir, dict_file)
    dest_file = os.path.join(args.dataset_dir, '{}.dict.jsonl'.format(args.source_lang))
    os.system('mv {} {}'.format(dict_file, dest_file))
    
    dict_file = 'dict.{}.txt'.format(args.target_lang)
    dict_file = os.path.join(args.dataset_dir, dict_file)
    dest_file = os.path.join(args.dataset_dir, '{}.dict.jsonl'.format(args.target_lang))
    os.system('mv {} {}'.format(dict_file, dest_file))
    
    
    def move_index_and_bin(split):
        bin = os.path.join(args.dataset_dir, '{}.{}-{}.{}.bin'.format(split, args.source_lang, args.target_lang, args.source_lang))
        idx = os.path.join(args.dataset_dir, '{}.{}-{}.{}.idx'.format(split, args.source_lang, args.target_lang, args.source_lang))
        os.system('mv {} {}'.format(idx, os.path.join(args.dataset_dir, '{}.{}.idx'.format(split, args.source_lang))))
        os.system('mv {} {}'.format(bin, os.path.join(args.dataset_dir, '{}.{}.bin'.format(split, args.source_lang))))
        
        bin = os.path.join(args.dataset_dir, '{}.{}-{}.{}.bin'.format(split, args.source_lang, args.target_lang, args.target_lang))
        idx = os.path.join(args.dataset_dir, '{}.{}-{}.{}.idx'.format(split, args.source_lang, args.target_lang, args.target_lang))
        os.system('mv {} {}'.format(idx, os.path.join(args.dataset_dir, '{}.{}.idx'.format(split, args.target_lang))))
        os.system('mv {} {}'.format(bin, os.path.join(args.dataset_dir, '{}.{}.bin'.format(split, args.target_lang))))

    move_index_and_bin('train')
    move_index_and_bin('test')
    move_index_and_bin('valid')

    