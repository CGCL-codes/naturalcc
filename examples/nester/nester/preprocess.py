
import json
from ast_operation import FunctionLocator, AnnotationCleaner, CommentRemover
import ast
import traceback
import random
from tqdm import tqdm
from ast_operation import AnnotationMask
import argparse




def get_function(jsonfile):
    data = json.loads(open(jsonfile, 'r', encoding = 'utf-8').read())
    locator = FunctionLocator()
    cleaner = AnnotationCleaner()
    remover = CommentRemover()
    sources = {}
    for d in tqdm(data):
        try:
            root = ast.parse(open(d["file"], "r").read())
            node = locator.run(root, d["loc"], d["name"], d["scope"])
            if node == None:
                print('{}--{}--{}--{}'.format(d["file"], d["loc"], d["name"], d["scope"]))
                print('Cannot find the function')
                exit()
            node = cleaner.run(node)
            node = remover.run(node)
            source = ast.unparse(node)
            sources['{}--{}--{}--{}'.format(d["file"], d["loc"], d["name"], d["scope"])] = source
        except Exception as e:
            print('{}--{}--{}--{}'.format(d["file"], d["loc"], d["name"], d["scope"]))
            print('Error Occurs, reason: {}'.format(e))
    with open(jsonfile.replace(".json", "_source.json"), "w", encoding = "utf-8") as jf:
        jf.write(json.dumps(sources, sort_keys=True, indent=4, separators=(',', ': ')))


def transform_dataset(dataset):
    instances = json.loads(open(dataset, 'r', encoding = 'utf-8').read())
    new_instances = {}
    cat = ["simple", "user-defined", "depth=1", "depth=0", "depth>=2"]
    for d in instances:
        category = None
        if d["cat"] in cat:
            category = d["cat"]
        if d["cat"] == "builtins" and not d["generic"]:
            category = "simple"
        if d["cat"] == "builtins" and d["generic"]:
            if d["type_depth"] == 0:
                category = "depth=0"
            elif d["type_depth"] == 1:
                category = "depth=1"
            elif d["type_depth"] >= 2:
                category = "depth>=2"
        new_instances['{}--{}--{}--{}'.format(d["file"], d["loc"], d["name"], d["scope"])] = [d['name'], d['processed_gttype'], category]
    
    with open(dataset.replace('.json', '_transformed.json'), 'w', encoding = 'utf-8') as df:
        df.write(json.dumps(new_instances, sort_keys=True, indent=4, separators=(',', ': ')))


def get_masked_function(test_file, res_file, mask):
    data = json.loads(open(test_file, 'r', encoding = 'utf-8').read())
    mask = AnnotationMask(mask = mask)
    sources = {}
    for d in tqdm(data):
        try:
            root = ast.parse(open(d["file"], "r").read())
            node = mask.run(root, d["name"], d["scope"], d["loc"])
            if node == None:
                print('{}--{}--{}--{}'.format(d["file"], d["loc"], d["name"], d["scope"]))
                print('Cannot generate the masked function')
                exit()
            source = ast.unparse(node)
            sources['{}--{}--{}--{}'.format(d["file"], d["loc"], d["name"], d["scope"])] = source
        except Exception as e:
            print('{}--{}--{}--{}'.format(d["file"], d["loc"], d["name"], d["scope"]))
            print('Error Occurs, reason: {}'.format(e))
            traceback.print_exc()
    
    with open(res_file, "w", encoding = "utf-8") as jf:
        jf.write(json.dumps(sources, sort_keys=True, indent=4, separators=(',', ': ')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required = True, type=str, help = "Path to the JSON source file")
    parser.add_argument('-r', '--res', required = False, type=str, help = "Path of output JSON file")
    parser.add_argument('-c', '--code', required = False, default = False, action = "store_true", help = "Extract the functions of instances")
    parser.add_argument('-t', '--transform', required = False, default = False, action = "store_true", help = "Transform the metadata of dataset")
    parser.add_argument('-m', '--mask', required = False, default = False, action = "store_true", help = "Get the masked functions of instances")
    args = parser.parse_args()
    if args.code:
        get_function(args.source)
    elif args.transform:
        transform_dataset(args.source)
    elif args.mask:
        get_masked_function(args.source, args.res, "<MASK>")

if __name__ == "__main__":
    main()