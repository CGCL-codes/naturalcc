from typing import Optional

import fire
import ast

from llama import Llama
import json
import re
import json
import os
from tqdm import tqdm

from collections import Counter


with open("/home/ligen/lg/codellama/data/testset_source_filter_high_level.json") as f:
    testset = json.load(f)
with open("/home/ligen/lg/llama3/NSTI_after_continue1_llama3_simple.json") as f:
    answer = json.load(f)

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.7,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    high_level = {}
    zero = 0
    for key in tqdm(answer.keys()):
        #zero = zero + 1
        #if zero == 100:
        #    break;
        parts = key.split('--')
        #print(parts[-2]) name
        #print(parts[-1])
        #exit(1)
        if parts[-1] == 'local':
#            zero = zero + 1
#            if zero == 3333:
#                break;
            try:
                instructions = [
                                    [
                                        {
                                            "role": "system",
                                            "content": "Assume you are a code converter. Your task is to transform Python code into high-level program descriptions line by line using specific transformation methods."
                                                       " Each line of Python code should be analyzed and transformed into a high-level program line that exactly corresponds to the code's functionality. Use only the following APIs: 'if_analysis()', 'assignment_analysis()'."
                                                       " Each transformation should use one of these APIs with the correct parameters directly derived from the code."
                                        },
                                        {
                                            "role": "user",
                                            "content":
                                                "Code:" + "urlpatterns = [path('about_project', views.index, name='video')]" + "\n" +
                                                "High-level Program: urlpatterns = Assignment_analyasis([path('about_project', views.index, name='video')])" + "\n" +
                                                "Code:" + "buf = self.getvalue()" + "\n" +
                                                "High-level Program: buf = Assignment_analyasis(self.getvalue())" + "\n" +
                                                "Code:" + "If val is None:\napp_name = 'about'\n" + "\n" +
                                                "High-level Program: If_analysis(val, None, is)\n app_name = Assignment_analysis('about')"+ "\n" +
                                                "Code:" + "If val is None:\na = 123\nelse:\na = 'abc'" + "\n" +
                                                "High-level Program:If_analysis(val, None, is)\na = Assignment_analysis(123)\nIf_analysis(val, None, is not)\na = Assignment_analysis('abc')" + "\n" +
                                                "Code:" + "metrics_to_return = {}\n        if not self.error_analysis and name.startswith('_'):\n        if is_empty_metric(metric):\n        if isinstance(metric, CategoricalAccuracy):" + "\n" +
                                                "High-level Program:metrics_to_return = Assignment_analysis({}) If_analysis(not self.error_analysis and name.startswith('_')) If_analysis(is_empty_metric(metric)) If_analysis(isinstance(metric, CategoricalAccuracy))" + "\n" +
                                                "Code:" + testset[key] + "\n" +
                                                "High-level Program:"+ "[to be generated High-level Program using specified APIs]"

                                        }
                                    ],
                                ]

                results = generator.chat_completion(
                    instructions,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                high_level[key] = results[0]['generation']['content']
            except:
                high_level[key] = parts[-2] + "= Assignment_analyasis()"
                pass;
        elif parts[-1] == 'arg':
            continue
            #zero = zero + 1
            #if zero == 3333:
             #   break;
            try:
                instructions = [
                    [
                        {
                            "role": "system",
                            "content": "Assume you are a code converter. Your task is to transform Python code into high-level program descriptions line by line using specific transformation methods."
                                        " Each line of Python code should be analyzed and transformed into a high-level program line that exactly corresponds to the code's functionality. Use only the following APIs: 'Fuction_Analysis', 'If_Analysis()', 'Argument_Analysis()'."
                                        " Each transformation should use one of these APIs with the correct parameters directly derived from the code."
                        },
                        {
                            "role": "user",
                            "content":

                                "Code:" + "def boot(self, container):\n provider = container.get(settings.Props.DI_PROVIDER)" + "\n" +
                                "High-level Program: Fuction_Analysis(boot(self, container))\n Argument_Analysis(provider = container.get(settings.Props.DI_PROVIDER))" + "\n" +
                                "Code:" + "def __init__(self, value):\n self.value = value" + "\n" +
                                "High-level Program: Fuction_Analysis(__init__(self, value))\n Argument_Analysis(self.value = value)" + "\n" +
                                "Code:" + "def _test_convenience_model_restorer(restorer, convenience_method, placeholder_model, trained_model, ckpt_id, capsys):\n _check_log(restorer, ckpt_id, capsys)" + "\n" +
                                "High-level Program: Fuction_Analysis(_test_convenience_model_restorer(restorer, convenience_method, placeholder_model, trained_model, ckpt_id, capsys))\n Argument_Analysis(_check_log(restorer, ckpt_id, capsys))" + "\n" +
                                "Code:" + "def get_mods_manifest(manifest_url):\n if n == 1:\n return json.loads(get_requests_object(manifest_url).text)\n if n == 1:" + "\n" +
                                "High-level Program: Fuction_Analysis(get_mods_manifest(manifest_url))\n If_Analysis(n==1)\n Argument_Analysis(return json.loads(get_requests_object(manifest_url).text))\n" + "\n" +
                                "Code:" + "def save_modlines(manifest_url, mods_details, mods_path):\n if mod_ids.difference(set(mods_details)):\n modlines[modline] = [mods_details[mod_id]['directory_name'] for mod_id in mod_ids]" + "\n" +
                                "High-level Program: Fuction_Analysis(save_modlines(manifest_url, mods_details, mods_path))\n If_Analysis(mod_ids.difference(set(mods_details)))\n Argument_Analysis(modlines[modline] = [mods_details[mod_id]['directory_name'] for mod_id in mod_ids])" + "\n" +
                                "Code:" + testset[key][0] + "\n" +

                                "High-level Program:"+ "[to be generated High-level Program using specified APIs]"
                        }
                    ],
                ]


                results = generator.chat_completion(
                    instructions,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                high_level[key] = results[0]['generation']['content']
            except:
                high_level[key] = "no answer"
                pass;
        elif parts[-1] == 'return':
            continue
            try:
                instructions = [
                    [
                        {
                            "role": "system",
                            "content": "Assume you are a code converter. Your task is to transform Python code into high-level program descriptions line by line using specific transformation methods."
                                       " Each line of Python code should be analyzed and transformed into a high-level program line that exactly corresponds to the code's functionality. Use only the following APIs: 'If_Analysis()', 'Return_Analysis()', 'Combine()'."
                                       " Each transformation should use one of these APIs with the correct parameters directly derived from the code."
                        },
                        {
                            "role": "user",
                            "content":

                                "Code:" + "return TestApp(app())" + "\n" +
                                "High-level Program: Return_Type1 = Return_Analysis(TestApp(app())) Return_Type = Return_Type1" + "\n" +
                                "Code:" + "return json.loads(get_requests_object(manifest_url).text)" + "\n" +
                                "High-level Program: Return_Type1 = Return_Analysis(json.loads(get_requests_object(manifest_url).text)) Return_Type = Return_Type1" + "\n" +
                                "Code:" + "if n == 1:\n    return mods_details" + "\n" +
                                "High-level Program: If_Analysis(n == 1)\n Return_Type1 = Return_Analysis(mods_details) Return_Type = Return_Type1" + "\n" +
                                "Code:" + "if code != 0:\n  return False\n  if counter == 3:\n return True" + "\n" +
                                "High-level Program: If_Analysis(code != 0)\n Return_Type1 = Return_Analysis(False)\n If_Analysis(counter == 3)\n Return_Type2 = Return_Analysis(True) Return_Type = Combine(Return_Type1, Return_Type2)" + "\n" +
                                "Code:" + "if payload.get('type') == 'auth':\n return json_success({'full_name': user_profile.full_name, 'email': user_profile.email, 'id': user_profile.id})\n if topic is None:\n if topic is None:\n if content is None:\n return json_success()" + "\n" +
                                "High-level Program: If_Analysis(payload.get('type') == 'auth')\n Return_Type1 = Return_Analysis(json_success({'full_name': user_profile.full_name, 'email': user_profile.email, 'id': user_profile.id}))\n If_Analysis(topic is None)\n If_Analysis(topic is None)\n If_Analysis(content is None)\n Return_Type2 = Return_Analysis(json_success()) Return_Type = Combine(Return_Type1, Return_Type2)" + "\n" +
                                "Code:" + testset[key] + "\n" +

                                "High-level Program:"+ "[to be generated High-level Program using specified APIs]"

                        }
                    ],
                ]

                results = generator.chat_completion(
                    instructions,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                high_level[key] = results[0]['generation']['content']
            except:
                high_level[key] = "no answer"
                pass;

    output_json_file = "/home/ligen/lg/codellama/high_level_local_llama3.json"

    with open(output_json_file, "w") as json_file:
        json.dump(high_level, json_file, indent=2)
    print(f"Results have been written to {output_json_file}.")


if __name__ == "__main__":
    fire.Fire(main)
