import json
from hityper.typeobject import TypeObject
import csv
import re, os
import argparse
import json
from tqdm import tqdm
def match_type_for_cot(string):
    pattern = re.compile(r'\`[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*\`')
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'\`[a-zA-Z\.\,\[\] ]+\`')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return None
        else:
            res = second_matched[-1].replace("`", "").replace('NoneType', 'None')#.replace("is ", "")
            if (" " in res and "[" not in res) or res.lower() == "unknown":
                res = None
            return res
    else:
        res = matched[-1].replace("`", "").replace('NoneType', 'None')#.replace("is ", "")
        if (" " in res and "[" not in res) or res.lower() == "unknown":
            res = None
        return res

#rules
with open("./NSTI_after_continue1_llama3.json") as f:
    NSTI_res_unpreprocessed = json.load(f)
NSTI_dedundancy1_simple = {}
for key in tqdm(NSTI_res_unpreprocessed.keys()):
    try:
        a = match_type_for_cot(NSTI_res_unpreprocessed[key][0])
        NSTI_dedundancy1_simple[key] = a
    except:
        NSTI_dedundancy1_simple[key] = ""


output_json_file = "./NSTI_after_continue1_llama3_simple.json"

with open(output_json_file, "w") as json_file:
    json.dump(NSTI_dedundancy1_simple, json_file, indent=2)
print(f"Results have been written to {output_json_file}.")
