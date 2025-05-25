import numpy as np
import argparse
from collections import Counter
import jieba
import json
from tqdm import tqdm
import numpy as np
from fastbm25 import fastbm25
from collections import defaultdict
from multiprocessing import Pool
import os
from config import datafiles


def partition_arg_topK(matrix, K, axis=1):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(-matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]



    
def get_corpus(train_source_file):
    cnt = 0
    with open(train_source_file, 'r') as f:
        trainset = json.load(f)
    train_code = []
    for k,v in trainset.items():

        k = k.split('--')
        train_code.append(v + ' ' + k[-2])
        
    print("building tokenization ...")
    
    new_train_code = []
    for c in tqdm(train_code):
        tokens = jieba.cut(c)
        tokens = [t for t in tokens if t != ' ']
        new_train_code.append(tokens)
    
    return new_train_code


def _get_similar_test(args):
    bm25_model = args[0]
    source = args[1]
    item = args[2]
    
    
    print(key)

    return [key, tops]

def get_similar_test(corpus, test_file, test_source_file, topk_file, K = 20):
    with open(test_file, 'r') as f:
        test_sample = json.load(f)
    with open(test_source_file) as f:
        source = json.load(f)

    topk_pair = defaultdict(list)
    bm25_model = fastbm25(corpus)

    for item in tqdm(test_sample):
        key = "{}--{}--{}--{}".format(item['file'], item['loc'], item['name'], item['scope'])
        if key in topk_pair:
            continue
        topk_pair[key] = []
        code = source[key] + ' ' + item['name']
        
        code_token = list(jieba.cut(code))
        code_token = [t for t in code_token if t!=' ']
        
        topk_results = bm25_model.top_k_sentence(code_token, k=K)
        
        for i in range(K):
            topk_pair[key].append(corpus.index(topk_results[i][0]))
    
    with open(topk_file, 'w') as f:
        f.write(json.dumps(topk_pair, indent=6))
       

def construct_incontext_data(train_file, train_source_file, topk_file, topk_with_label_file):
    def search(trainset, train_key):
        train_key = train_key.split('--')
        if len(train_key) > 4:
            scope = train_key[-1]
            name = train_key[-2]
            loc = train_key[-3]
            file = '--'.join(train_key[:-3])
        else:
            file, loc, name, scope = train_key
        for item in trainset:
            if item['file'] == file and item['loc'] == loc and item['name'] == name and item['scope'] == scope:
                return item['processed_gttype']
    with open(similar_file) as f:
        topk = json.loads(f.read())
    with open(train_file) as f:
        trainset = json.load(f) 
    with open(train_source_file) as f:
        train_source = json.load(f) 
    new_dict = []
    train_keys, train_code = list(train_source.keys()), list(train_source.values())
    for test_k, v in tqdm(topk.items()):
        item_list = []
        for idx in v:
            similar_key = train_keys[idx]
            label = search(trainset, similar_key)
            item_list.append({similar_key : label})
        new_dict.append({test_k:item_list})
    
    newdata = {}
    for d in new_dict:
        for c in d:
            newdata[c] = d[c]
    
    with open(topk_with_label_file, 'w') as f:
        json.dump(newdata, f, indent = 6)
        
def gen_topk(train_file, train_source_file, test_file, test_source_file, topk_file, topk_with_label_file, K = 20):
    # train_file, train_source_file, test_file, test_source_file are required input files
    # topk_file is an intermediate file and you can set it to any path
    # topk_with_label_file is the final output file required in our approach
    print("getting corpus ... ")
    corpus = get_corpus(train_source_file)
    print("Finding Top K  ... ")
    get_similar_test(corpus, test_file, test_source_file, topk_file, K = K)
    construct_incontext_data(train_file, train_source_file, topk_file, topk_with_label_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', default = 'data', type=str, help = "Path to the folder of source files")
    parser.add_argument('-k', '--topk', default = 20, required = False, type=int, help = "Top K similar demonstrations")
    parser.add_argument('-o', '--no_slice', default = False, required = False, action = "store_true", help = "Use no sliced code")
    parser.add_argument('-p', '--hop', default = 3, type=int, required = False, help = "Number of hops")
    args = parser.parse_args()

    if args.no_slice:
        train_source_file = os.path.join(args.source, datafiles["trainset_sourcecode"])
        test_source_file = os.path.join(args.source, datafiles["testset_sourcecode"])
        topk_with_label_file = os.path.join(args.source, datafiles["similar_demos"])
    else:
        train_source_file = os.path.join(args.source, datafiles["trainset_sliced_sourcecode"].replace("HOP", str(args.hop)))
        test_source_file = os.path.join(args.source, datafiles["testset_sliced_sourcecode"].replace("HOP", str(args.hop)))
        topk_with_label_file = os.path.join(args.source, datafiles["similar_sliced_demos"].replace("HOP", str(args.hop)))

    gen_topk(
        os.path.join(args.source, datafiles["trainset_metadata"]),
        train_source_file,
        os.path.join(args.source, datafiles["testset_metadata"]),
        test_source_file,
        os.path.join(args.source, "topk_pair.json"),
        topk_with_label_file,
        K = args.topk
    )

if __name__ == "__main__":
    main()
    
