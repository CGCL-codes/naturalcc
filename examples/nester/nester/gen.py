import json
from ast_operation import AnnotationMask
from tqdm import tqdm
import ast
import traceback
from transformers import RobertaTokenizer, RobertaForMaskedLM, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from unixcoder import UniXcoder
import re
from incoder import infill
from typegen import construct_cot_prompt, construct_regular_prompt
import os
import argparse
from config import datafiles, cache_dir


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def run(resfile,
        model, 
        data_folder,
        demo_num = 6, 
        usertype = True, 
        hop = 3, 
        fix_demo = False,
        cot = True,
        slice_code = True,
    ):
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir = cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model, cache_dir = cache_dir, pad_token_id = tokenizer.eos_token_id).to(DEVICE)
    with open(os.path.join(data_folder, datafiles["testset_metadata"])) as f:
        testset = json.load(f)
    if slice_code:
        with open(os.path.join(data_folder, datafiles["testset_sliced_sourcecode"].replace("HOP", str(hop)))) as f:
            test_source = json.load(f)
        with open(os.path.join(data_folder, datafiles["trainset_sliced_sourcecode"].replace("HOP", str(hop)))) as f:
            train_source = json.load(f)
    else:
        with open(os.path.join(data_folder, datafiles["testset_sourcecode"])) as f:
            test_source = json.load(f)
        with open(os.path.join(data_folder, datafiles["trainset_sourcecode"])) as f:
            train_source = json.load(f)

    with open(os.path.join(data_folder, datafiles["testset_usertypes"])) as f:
        test_usertypes = json.load(f)
    with open(os.path.join(data_folder, datafiles["trainset_usertypes"])) as f:
        train_usertypes = json.load(f)
    with open(os.path.join(data_folder, datafiles["transformed_trainset_metadata"])) as f:
        trainset = json.load(f)

    if fix_demo:
        with open(os.path.join(data_folder, datafiles["fixed_examples"])) as f:
            fixed_inc = json.load(f)
    else:
        if slice_code:
            with open(os.path.join(data_folder, datafiles["similar_sliced_domos"].replace("HOP", str(hop)))) as f:
                topk = json.load(f)
        else:
            with open(os.path.join(data_folder, datafiles["similar_domos"])) as f:
                topk = json.load(f)
    if cot:
        with open(os.path.join(data_folder, datafiles["trainset_cots"].replace("HOP", str(hop)))) as f:
            cots = json.load(f)

    # Fixed demostrations
    if fix_demo:
        inc_prompt = ""
        for item in fixed_inc[:demo_num]:
            key = "{}--{}--{}--{}".format(item['file'], item['loc'], item['name'], item['scope'])
            code = train_source[key]
            if cot:
                train_item = item
                train_item["cot"] = cots[key]
                if usertype:
                    inc_prompt += construct_cot_prompt(train_item, code, usertypes = train_usertypes[key][1], mode = "Auto")
                else:
                    inc_prompt += construct_cot_prompt(train_item, code, mode = "Auto")
            else:
                if usertype:
                    inc_prompt += construct_regular_prompt(item, code, usertypes = train_usertypes[key][1])
                else:
                    inc_prompt += construct_regular_prompt(item, code)
    
    
    predictions = {}
    for item in tqdm(testset):
        key = "{}--{}--{}--{}".format(item['file'], item['loc'], item['name'], item['scope'])
        current_code = test_source[key]
        
        if not fix_demo:
            inc_prompt = ""
            if demo_num > 0:
                inc = topk[key][:demo_num]
                inc.reverse()
                inc_prompt = ""
                inc_prompts = []
                for c in inc:
                    code = train_source[list(c.keys())[0]]
                    name, gttype = trainset[list(c.keys())[0]]
                    if cot:
                        train_item = {"name": name, "processed_gttype": gttype, "scope": list(c.keys())[0].split("--")[-1], "cot": cots[list(c.keys())[0]]}
                    else:
                        train_item = {"name": name, "processed_gttype": gttype, "scope": list(c.keys())[0].split("--")[-1]}
                    if usertype:
                        if cot:
                            inc_prompt = inc_prompt + construct_cot_prompt(train_item, code, usertypes = train_usertypes[list(c.keys())[0]][1], mode = "Auto")
                        else:
                            inc_prompt = inc_prompt + construct_regular_prompt(train_item, code, usertypes = train_usertypes[list(c.keys())[0]][1])
                    else:
                        if cot:
                            inc_prompt = inc_prompt + construct_cot_prompt(train_item, code, mode = "Auto")
                        else:
                            inc_prompt = inc_prompt + construct_regular_prompt(train_item, code)
        
        prompt_str = ''
        prompt_str += inc_prompt
        if usertype:
            if cot:
                question_str = construct_cot_prompt(item, current_code, omit_type = True, usertypes = test_usertypes[key][1], mode = "Auto")
            else:
                question_str = construct_regular_prompt(item, current_code, omit_type = True, usertypes = test_usertypes[key][1])
        else:
            if cot:
                question_str = construct_cot_prompt(item, current_code, omit_type = True, mode = "Auto")
            else:
                question_str = construct_regular_prompt(item, current_code, omit_type = True)
        prompt_str = prompt_str + question_str
        inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        size = input_ids.size()[1]
        if size + 128 > 2048:
            predictions[key] = []
            print("Overlong context, skipped.")
            continue
        else:
            try:
                outputs = model.generate(input_ids, attention_mask = attention_mask, do_sample = False, num_beams = 5, num_return_sequences = 5, max_length = size + 128)
                prediction = tokenizer.batch_decode(outputs)
                prediction = [pred[len(prompt_str): ] for pred in prediction]
                predictions[key] = prediction
            except Exception as e:
                print(e)
                predictions[key] = []
        
       
    with open(resfile, "w", encoding = "utf-8") as cf:
        cf.write(json.dumps(predictions, sort_keys=True, indent=4, separators=(',', ': ')))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', default = "data", type=str, help = "Path to the folder of source files")
    parser.add_argument('-r', '--res', required = True, type=str, help = "Path of output JSON file")
    parser.add_argument('-m', '--model', required = True, type=str, help = "Name of pre-trained model in HuggingFace")
    parser.add_argument('-f', '--fix_demo', required = False, default = False, action="store_true", help = "Use fixed demonstrations")
    parser.add_argument("-o", '--no_slice', required = False, default = False, action = "store_true", help = "Do not use code slices")
    parser.add_argument('-c', '--remove_cot', required = False, default = False, action="store_true", help = "Remove COT prompts")
    parser.add_argument('-u', '--remove_usertypes', required = False, default = False, action = "store_true", help = "Remove user type hints")
    parser.add_argument('-n', '--demo_num', required = False, default = 5, type=int, help = "Number of demonstrations")
    parser.add_argument('-p', '--hop', required = False, default = 3, type=int, help = "Number of hops")
    args = parser.parse_args()

    run(args.res, args.model, args.source, demo_num = args.demo_num, usertype = False if args.remove_usertypes else True, hop = args.hop, fix_demo = args.fix_demo, cot = False if args.remove_cot else True, slice_code = False if args.no_slice else True)

if __name__ == "__main__":
    main()


