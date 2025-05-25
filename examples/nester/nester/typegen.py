import os
import openai
import json
from tqdm import tqdm
import time
import timeout_decorator
import base64
import traceback
import argparse
from config import openai_keys, max_retry, error_sleep, normal_sleep, datafiles, chatgpt, gpt3

api_index = 0


def construct_regular_prompt(item, ori_code, omit_type = False, usertypes = None):
    maps = {
        "arg": "the argument",
        "return": "the return value of",
        "local": "variable"
    }
    if not omit_type:
        if usertypes != None:
            prompt_str = "Python code: \n {}\nAvailable user-defined types:\n {}\nQ: What is the type of {} {}?\nA: The type of {} {} is `{}`.\n".format(ori_code, ", ".join(usertypes), maps[item['scope']], item['name'], maps[item['scope']], item['name'], item['processed_gttype'])
        else:
            prompt_str = "Python code: \n {}\nQ: What is the type of {} {}?\nA: The type of {} {} is `{}`.\n".format(ori_code, maps[item['scope']], item['name'], maps[item['scope']], item['name'], item['processed_gttype'])
    else:
        if usertypes != None:
            prompt_str = "Python code: \n {}\nAvailable user-defined types:\n {}\nQ: What is the type of {} {}?\nA: The type of {} {} is ".format(ori_code, ", ".join(usertypes), maps[item['scope']], item['name'], maps[item['scope']], item['name'])
        else:
            prompt_str = "Python code: \n {}\nQ: What is the type of {} {}?\nA: The type of {} {} is ".format(ori_code, maps[item['scope']], item['name'], maps[item['scope']], item['name'])
    return prompt_str


def construct_cot_prompt(item, ori_code, omit_type = False, usertypes = None, mode = "Self"):
    maps = {
        "arg": "the argument",
        "return": "the return value of",
        "local": "variable"
    }
    if not omit_type:
        if usertypes != None:
            prompt_str = "Python code: \n {}\nAvailable user-defined types:\n {}\nQ: What is the type of {} {}?\n".format(ori_code, ", ".join(usertypes), maps[item['scope']], item['name'])
        else:
            prompt_str = "Python code: \n {}\nQ: What is the type of {} {}?\n".format(ori_code, maps[item['scope']], item['name'])
        if mode == "Self":
            prompt_str += "A: Let's think step by step.\n"
        elif mode == "Auto":
            prompt_str += "A: {}\n".format(item["cot"])
    else:
        if usertypes != None:
            prompt_str = "Python code: \n {}\nAvailable user-defined types:\n {}\nQ: What is the type of {} {}?\n".format(ori_code, ", ".join(usertypes), maps[item['scope']], item['name'])
        else:
            prompt_str = "Python code: \n {}\nQ: What is the type of {} {}?\n".format(ori_code, maps[item['scope']], item['name'])
        prompt_str += "A: "
    return prompt_str


# Timeout the query when it takes longer than 60s, the default timeout in request is too long (600s)
@timeout_decorator.timeout(60)
def run_chatgpt(messages):
    return openai.ChatCompletion.create(
            model=chatgpt,
            messages = messages,
            temperature=1,
            max_tokens=128,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n = 50)


@timeout_decorator.timeout(60)
def run_gpt3(prompt_str):
    return openai.Completion.create(
            model=gpt3,
            prompt=prompt_str,
            temperature=1,
            max_tokens=128,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n = 25
            )

def run(data_folder, resfile, model, demo_num = 5, usertype = True, hop = 3, api_index = api_index, cont = False, fix_demo = False, cot = True, slice_code = True):
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
    
    
    try:
        if cont:
            with open(resfile, "r") as rf:
                predictions = json.loads(rf.read())
        else:
            predictions = {}
        for item in tqdm(testset):
            key = "{}--{}--{}--{}".format(item['file'], item['loc'], item['name'], item['scope'])
            current_code = test_source[key]
            if key in predictions and len(predictions[key]) > 0:
                continue

            # Demonstrations based on BM25
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
            
            # Construct question prompt
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
            Flag_no_error = True
            num = 0
            if model == "chatgpt":
                messages = [{"role": "user", "content": prompt_str}]
            while Flag_no_error and num < max_retry:
                try:
                    if model == "chatgpt":
                        response = run_chatgpt(messages)
                    else:
                        response = run_gpt3(prompt_str)
                    Flag_no_error = False
                    num += 1
                    time.sleep(normal_sleep)
                    preds = []
                    if model == "chatgpt":
                        for c in response["choices"]:
                            preds.append(c["message"]["content"])
                    else:
                        for c in response["choices"]:
                            preds.append(c["text"])
                    predictions[key] = preds
                except Exception as e:
                    print(e)
                    if 'Please reduce the length of the messages' in str(e) or 'Please reduce your prompt' in str(e):
                        predictions[key] = []
                        break
                    if 'You exceeded your current quota' in str(e):
                        from apikeys import keys
                        api_index += 1
                        openai.api_key = keys[api_index]
                        num -= 1
                    time.sleep(error_sleep)
                    Flag_no_error = True
                    num += 1
                    if key not in predictions:
                        predictions[key] = []
                    continue
        with open(resfile, "w", encoding = "utf-8") as cf:
            cf.write(json.dumps(predictions, sort_keys=True, indent=4, separators=(',', ': ')))
    except Exception as e:
        print("Error Occurs, reason: {}".format(e))
        traceback.print_exc()
        with open(resfile.replace(".json", "_ERROR.json"), "w", encoding = "utf-8") as cf:
            cf.write(json.dumps(predictions, sort_keys=True, indent=4, separators=(',', ': ')))
    except KeyboardInterrupt:
        with open(resfile.replace(".json", "_Canceled.json"), "w", encoding = "utf-8") as cf:
            cf.write(json.dumps(predictions, sort_keys=True, indent=4, separators=(',', ': ')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', default = "data", type=str, help = "Path to the folder of source files")
    parser.add_argument('-r', '--res', required = True, type=str, help = "Path of output JSON file")
    parser.add_argument('-m', '--model', required = False, default = 'chatgpt', type=str, help = "Model used in the inference, choose from chatgpt and gpt3")
    parser.add_argument('-f', '--fix_demo', required = False, default = False, action="store_true", help = "Use fixed demonstrations")
    parser.add_argument("-o", '--no_slice', required = False, default = False, action = "store_true", help = "Do not use code slices")
    parser.add_argument('-c', '--remove_cot', required = False, default = False, action="store_true", help = "Remove COT prompts")
    parser.add_argument('-u', '--remove_usertypes', required = False, default = False, action = "store_true", help = "Remove user type hints")
    parser.add_argument('-i', '--incremental', required = False, default = False, action = "store_true", help = "Incremental inference based on the existing result file")
    parser.add_argument('-n', '--demo_num', required = False, default = 5, type=int, help = "Number of demonstrations")
    parser.add_argument('-p', '--hop', required = False, default = 3, type=int, help = "Number of hops")
    args = parser.parse_args()
    try:
        if openai_keys[0] == "key1":
            raise ValueError()
        openai.api_key = openai_keys[api_index]
    except:
        print("Error: please input the OpenAI keys in config.py first.")
        exit()

    run(args.source, args.res, args.model, demo_num = args.demo_num, usertype = False if args.remove_usertypes else True, hop = args.hop, cont = args.incremental, fix_demo = args.fix_demo, cot = False if args.remove_cot else True, slice_code = False if args.no_slice else True)

    

        
if __name__ == "__main__":
    main()
