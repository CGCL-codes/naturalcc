import json
from hityper.typeobject import TypeObject
import csv
import re, os
import argparse


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

def match_type(string):
    string = string.split("\nPython Code:")[0].split("\nQ:")[0]
    pattern = re.compile(r'\`[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*\`')
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'\`[a-zA-Z\.\,\[\] ]+\`')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return string.split("\n")[0][:-1]
        else:
            return second_matched[0].replace("`", "")
    else:
        return matched[0].replace("`", "")

def match_type_for_completion(string):
    string = string.split("\nPython Code:")[0].split("\nQ:")[0]
    pattern = re.compile(r'[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*')
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'[a-zA-Z\.\,\[\] ]+')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return string.split("\n")[0][:-1]
        else:
            return second_matched[0].replace("`", "")
    else:
        return matched[0].replace("`", "")


def extract_type_from_text(text):
    if len(text.split()) > 0:
        text = text.split()[0]
    else:
        text = text
    if text.endswith(".") or text.endswith(","):
        text = text[:-1]
    typeobjs = TypeObject.Str2Obj(text)
    return typeobjs


def extract_type_from_cot(text):
    text = text.split()[-1][:-1]
    typeobjs = TypeObject.Str2Obj(text)
    return typeobjs


def transform_sample_to_top(data, cot = False, case_sensitive = True):
    freq = {}
    for d in data:
        if cot:
            d = match_type_for_cot(d)
            if d == None:
                continue
        else:
            d = match_type(d)
            if d == None:
                continue
        found = None
        for k in freq:
            if k.lower() == d.lower() and not case_sensitive:
                found = k
                break
            elif k == d and case_sensitive:
                found = k
                break
        if found != None:
            freq[found] += 1
        else:
            freq[d] = 1
    
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    top = []
    for s in sorted_freq:
        top.append(s[0])
    return top




def evaluate(resfile, testset, detail = False, n = 5, batch = False, cot = False, sample = True, csv_file = None, exact = True, res_file = None, completion = False):
    results = json.loads(open(resfile, 'r').read())
    testset = json.loads(open(testset, 'r').read())
    reses = {}
    topn = {}
    empty = 0
    total = {
        "simple": 0,
        "user-defined": 0,
        "depth=1": 0,
        "depth=0": 0,
        "depth>=2": 0,
        "generic": 0,
        "arg": 0,
        "return": 0,
        "local": 0,
        "total": 0
    }
    for i in range(1, n+1):
        topn[i] = {
            "simple": 0,
            "user-defined": 0,
            "depth=1": 0,
            "depth=0": 0,
            "depth>=2": 0,
            "generic": 0,
            "arg": 0,
            "return": 0,
            "local": 0,
            "total": 0
        }
    for r in results:
        if r not in testset:
            empty += 1
            continue
        gttype = TypeObject.Str2Obj(testset[r][1])
        total["total"] += 1
        total[testset[r][2]] += 1
        total[r.split("--")[-1]] += 1
        if len(results[r]) == 0:
            empty += 1
            continue
        if testset[r][2].startswith("depth"):
            total["generic"] += 1
        if sample:
            predictions = transform_sample_to_top(results[r], cot = cot, case_sensitive = True)
        elif completion:
            predictions = results[r]
        else:
            if not cot:
                predictions = []
                for res in results[r]:
                    pred = match_type(res)
                    if pred == None:
                        predictions.append("invalid_type")
                    else:
                        predictions.append(pred)
            else:
                predictions = []
                for res in results[r]:
                    pred = match_type_for_cot(res.split('\nPython')[0])
                    if pred == None:
                        try:
                            predictions.append(res.split('\nPython')[0].split()[-1][:-1] if res.split('\nPython')[0].split()[-1].endswith(".") else res.split('\nPython')[0].split()[-1])
                        except:
                            predictions.append("invalid_type")
                    else:
                        predictions.append(pred)
        matched = False
        for index, pred in enumerate(predictions):
            if completion:
                pred_str = match_type_for_completion(pred)
                if pred_str != None:
                    predtype = TypeObject.Str2Obj(pred_str)
                else:
                    continue
            else:
                predtype = TypeObject.Str2Obj(pred)
            if (exact and TypeObject.isIdenticalSet(gttype, predtype)) or (not exact and TypeObject.isSetIncluded2(predtype, gttype)):
                matched = True
                reses[r] = [results[r], testset[r][1]]
                for i in range(index+1, n+1):
                    topn[i]["total"] += 1
                    topn[i][testset[r][2]] += 1
                    if testset[r][2].startswith("depth"):
                        topn[i]["generic"] += 1
                    topn[i][r.split("--")[-1]] += 1
                break
    if res_file != None:
        with open(res_file, "w", encoding = "utf-8") as rf:
            rf.write(json.dumps(reses, sort_keys=True, indent=4, separators=(',', ': ')))
    if not batch:
        print(f'Totally {len(results)} results, among them {empty} are empty.')
        print("Top-n acc:")
        for i in topn:
            print(f"top-{i}" , topn[i]["total"], topn[i]["total"] / total["total"])
        if detail:
            print("Detailed Acc:")
            for i in topn:
                print(f"top-{i}:")
                for c in topn[i]:
                    if c != "total":
                        print(c, topn[i][c], topn[i][c] / total[c])
        if csv_file:
            acc = [
                ["top-N", "overall", "simple", "user-defined", "depth=1", "depth=0", "depth>=2", "generic", "arg", "return", "local"]
            ]
            for i in topn:
                cur_acc = [
                    i, format(topn[i]["total"] / total["total"] * 100, ".1f")
                ]
                for c in topn[i]:
                    if c == "total":
                        continue
                    cur_acc.append(format(topn[i][c] / total[c] * 100, ".1f"))
                acc.append(cur_acc)
            with open(csv_file, "w", encoding = "utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerows(acc)
    else:
        return topn, total




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required = True, type=str, help = "Path to the JSON prediction file")
    parser.add_argument('-t', '--testset', required = True, type=str, help = "Path to the transformed testset metadata JSON file")
    parser.add_argument('-r', '--res', required = False, type=str, help = "Path of output CSV file, evaluation results will print to the screen if this option is not indicated")
    parser.add_argument('-m', '--sample', required = False, default = False, action = "store_true", help = "The predictions are sampled instead of top1-5, this option is required to evaluate TypeGen and other OpenAI models")
    parser.add_argument('-n', '--completion', required = False, default = False, action = "store_true", help = "The predictions are the completions of cloze-style approaches")
    parser.add_argument('-d', '--detail', required = False, default = False, action = "store_true", help = "Print detailed evaluation results")
    parser.add_argument('-c', '--cot', required = False, default = False, action = "store_true", help = "The predictions are in the format of COTs")
    parser.add_argument('-i', '--similar', required = False, default = False, action = "store_true", help = "Switch the evaluation metric to Match to Parametric")
    args = parser.parse_args()

    evaluate(args.source, args.testset, detail = args.detail, cot = args.cot, sample = args.sample, exact = True if not args.similar else False, csv_file = args.res if args.res else None, completion = args.completion)


if __name__ == "__main__":
    main()