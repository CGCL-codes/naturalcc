import json
from tqdm import tqdm
import ast
import traceback
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from unixcoder import UniXcoder
import re
from incoder import infill
from config import datafiles, cache_dir
import argparse

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def match_type(string):
    pattern = re.compile(r'[a-zA-Z\. ]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*')
    matched = re.match(pattern, string)
    if matched == None:
        return None
    else:
        return matched.group()


def build_prompt(source, tokenizer, max_len = 490):
    lines = source.split('\n')
    target_index = 0
    for index, l in enumerate(lines):
        if "<MASK>" in l:
            target_index = index
            break
    line_size = min(len(lines), 50)
    while line_size > 0:
        prompt = "\n".join(lines[target_index - line_size if target_index - line_size >=0 else 0 :target_index] + [lines[target_index]] + lines[target_index+1:target_index + line_size + 1])
        if tokenizer(prompt, return_tensors = "pt")["input_ids"].size()[1] < max_len:
            break
        line_size -= 1
    return prompt


def run_codet5(masked_source_file, res_file, model_name = "Salesforce/codet5-base"):
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir = cache_dir).to(DEVICE)
    tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
    sources = json.loads(open(masked_source_file, "r", encoding = "utf-8").read())
    predictions = {}
    for r in tqdm(sources):
        prompt = build_prompt(sources[r], tokenizer)
        inputs = tokenizer(prompt, return_tensors = "pt").to(DEVICE)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = model.generate(input_ids, num_beams = 50, num_return_sequences = 50)
        prediction = tokenizer.batch_decode(outputs)
        predictions[r] = prediction
    
    with open(res_file, "w", encoding = "utf-8") as cf:
        cf.write(json.dumps(predictions, sort_keys=True, indent=4, separators=(',', ': ')))

def run_unixcoder(masked_source_file, res_file, model_name = "microsoft/unixcoder-base"):
    model = UniXcoder(model_name, cache_dir).to(DEVICE)
    sources = json.loads(open(masked_source_file, "r", encoding = "utf-8").read())
    predictions = {}
    for r in tqdm(sources):
        source = sources[r].replace("<MASK>", "<mask0>")
        lines = source.split('\n')
        target_index = 0
        for index, l in enumerate(lines):
            if "<mask0>" in l:
                target_index = index
                break
        line_size = min(len(lines), 50)
        while line_size > 0:
            prompt = "\n".join(lines[target_index - line_size if target_index - line_size >=0 else 0 :target_index] + [lines[target_index]] + lines[target_index+1:target_index + line_size + 1])
            if len(model.tokenize([prompt], mode="<encoder-decoder>")) < 490:
                break
            line_size -= 1
        input_ids = torch.tensor(model.tokenize([prompt], max_length = 512, mode = "<encoder-decoder>")).to(DEVICE)
        outputs = model.generate(input_ids, decoder_only=False, beam_size=50, max_length=128)
        prediction = model.decode(outputs)
        prediction = [x.replace("<mask0>","").strip() for x in prediction[0]]
        new_prediction = []
        for p in prediction:
            t = match_type(p)
            if t != None:
                new_prediction.append(t)
        prediction = new_prediction
        predictions[r] = prediction
    with open(res_file, "w", encoding = "utf-8") as cf:
        cf.write(json.dumps(predictions, sort_keys=True, indent=4, separators=(',', ': ')))



def run_incoder(masked_source_file, res_file, mask = "<MASK>", BIG_MODEL = True):
    if BIG_MODEL:
        model_name = "facebook/incoder-6B"
        kwargs = dict(
            revision="float16", 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir = cache_dir
        )
    else:
        model_name = "facebook/incoder-1B"
        kwargs = dict(cache_dir = cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
    sources = json.loads(open(masked_source_file, "r", encoding = "utf-8").read())
    predictions = {}
    try:
        for r in tqdm(sources):
            prompt = build_prompt(sources[r], tokenizer, max_len = 1028)
            parts = prompt.split(mask)
            prediction = infill(parts, model, tokenizer)
            predictions[r] = prediction
    except Exception as e:
        print("Error Occurs: {}".format(e))
        traceback.print_exc()
    finally:
        with open(res_file, "w", encoding = "utf-8") as cf:
            cf.write(json.dumps(predictions, sort_keys=True, indent=4, separators=(',', ': ')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', default = "dataset", type=str, help = "Path to the folder of source files")
    parser.add_argument('-r', '--res', required = True, type=str, help = "Path of output JSON file")
    parser.add_argument('-m', '--model', required = True, type=str, help = "Name of pre-trained model in HuggingFace")
    args = parser.parse_args()
    if "codet5" in args.model:
        run_codet5(os.path.join(args.source, datafiles["testset_codet5_masked_code"]), args.res, model_name = args.model)
    elif "unixcoder" in args.model:
        run_unixcoder(os.path.join(args.source, datafiles["testset_regular_masked_code"]), args.res, model_name = args.model)
    elif "incoder" in args.model:
        if "1B" in args.model:
            run_unixcoder(os.path.join(args.source, datafiles["testset_regular_masked_code"]), args.res, BIG_MODEL = False)
        else:
            run_unixcoder(os.path.join(args.source, datafiles["testset_regular_masked_code"]), args.res, BIG_MODEL = True)


if __name__ == "__main__":
    main()

        


    



