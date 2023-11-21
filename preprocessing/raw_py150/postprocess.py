import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser("post process for raw_py150")
    parser.add_argument('-d', '--dataset-dir', default='data-bin/raw_py150', 
                        help='directory where your binary dataset locates')
    parser.add_argument('-l', '--language', default='code_tokens', help='should be code_tokens')
    args = parser.parse_args()
    
    dict_file = 'dict.{}.txt'.format(args.language)
    dict_file = os.path.join(args.dataset_dir, dict_file)
    dest_file = os.path.join(args.dataset_dir, 'dict.txt')
    os.system('mv {} {}'.format(dict_file, dest_file))
    
    train_bin = os.path.join(args.dataset_dir, 'train.code_tokens-None.{}.bin'.format(args.language))
    train_idx = os.path.join(args.dataset_dir, 'train.code_tokens-None.{}.idx'.format(args.language))
    os.system('mv {} {}'.format(train_bin, os.path.join(args.dataset_dir, 'train.bin')))
    os.system('mv {} {}'.format(train_idx, os.path.join(args.dataset_dir, 'train.idx')))
    
    test_bin = os.path.join(args.dataset_dir, 'test.code_tokens-None.{}.bin'.format(args.language))
    test_idx = os.path.join(args.dataset_dir, 'test.code_tokens-None.{}.idx'.format(args.language))
    os.system('mv {} {}'.format(test_bin, os.path.join(args.dataset_dir, 'test.bin')))
    os.system('mv {} {}'.format(test_idx, os.path.join(args.dataset_dir, 'test.idx')))