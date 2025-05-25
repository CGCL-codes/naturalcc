import json
from ast_operation import StaticSlicer
import ast
import traceback
import random
from tqdm import tqdm
import argparse

        
        
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

def static_slice(dataset, hop = 3):
    data = json.loads(open(dataset, "r").read())
    newdata = {}
    i = 0
    for r in tqdm(data):
        key = '{}--{}--{}--{}'.format(r["file"], r["loc"], r["name"], r["scope"])
        i += 1
        name = r["name"]
        scope = r["scope"]
        loc = r["loc"]
        filename = r["file"]
        slicer = StaticSlicer()
        newdata[key] = slicer.run(filename, name, scope, loc, forward_hop = hop)

    with open(dataset.replace(".json", f"_staticsliced_hop{hop}.json"), "w", encoding = "utf-8") as df:
        df.write(json.dumps(newdata, sort_keys=True, indent=4, separators=(',', ': ')))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required = True, type=str, help = "Path to the JSON file of dataset")
    parser.add_argument('-p', '--hop', required = False, default = 3, type=int, help = "Number of hops")
    args = parser.parse_args()
    static_slice(args.source, hop = args.hop)

if __name__ == "__main__":
    main()




