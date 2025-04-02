import os.path
import tqdm
from utils import *
import Levenshtein
import fnmatch

_ALT_ANS_FILE = 'annotations/alt_answers.jsonlines'
_ENABLING_ALT_ANSWERS = False
_INCORRECT_ANS_FILE = 'annotations/incorrect_answers.jsonlines'
_END_PREDICTION_SEP = '[END OF CODE]'
def read_alternative_answers(filename=_ALT_ANS_FILE):
    with open(filename) as f:
        lines = f.readlines()
    alt_ans_dict = {}
    for l in lines:
        d = json.loads(l)
        if d['qas_id'] not in alt_ans_dict:
            alt_ans_dict[d['qas_id']] = []
        if d['answer'] not in alt_ans_dict[d['qas_id']]:
            alt_ans_dict[d['qas_id']].append(d['answer'])

    return alt_ans_dict

def read_incorrect_answers(filename=_INCORRECT_ANS_FILE):
    return read_alternative_answers(filename=filename)

def read_manual_rationale(filename):
    with open(filename) as f:
        lines = f.readlines()
    d = {}
    for l in lines:
        id, rat = l.rstrip().split('\t')
        d[id] = rat
    return d
    
def read_example_data(fname, split, n_dist = -1, par_connection=' ', manual_annotation_style=None):
    data = read_json(fname)
    examples = []
    for d in data:
        ex = {
            "id": d["prob_id"],
            "problem": f"Write a function.\nFunction name: {d['name']}\nFunction signature:  {d['signature']}\nFunction Description:\n{d['problem']}",
            "rationale": [("# " if split == "python" else "// ") + str(idx+1) + ". " + x.strip() for idx, x in enumerate(d["rationales"])],
            "solution": d["solution"] + "\n" + _END_PREDICTION_SEP
        }
        examples.append(ex)
    return examples

def read_essential_external_data(fname, split, n_dist = -1, par_connection=' ', manual_annotation_style=None):
    data = read_json(fname)
    examples = []
    for d in data:
        ex = {
            "problem": f"Identify the externally-defined symbols (from a private library, unknown to you) that are essential to the task. The task is: Write a function.\nFunction name: {d['name']}\nFunction signature:  {d['signature']}\nFunction Description:\n{d['problem']}\nContext:\n{d['context']}",
            "solution": d["essential_externals"] + "\n" + _END_PREDICTION_SEP
        }
        examples.append(ex)
    return examples

def similarity(str1, str2):
    # Calculate Levenshtein distance
    distance = Levenshtein.distance(str1, str2)

    # Normalize the distance to be between 0 and 1. 0 means identical strings, 1 means completely different.
    # The max possible distance is the length of the longer string
    max_len = max(len(str1), len(str2))
    similarity_score = 1 - (distance / max_len)

    return similarity_score

def find_files(directory, patterns):
    for root, dirs, files in os.walk(directory):
        for pattern in patterns:
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename

import re

def extract_java_function_signature(text):
    """
    Extracts the function signature from a Java code snippet.

    Args:
    text (str): The Java code as a string.

    Returns:
    str: The function signature.
    """
    # Initial position pointers
    begin = 0
    current = 0

    # Keywords that could precede a function signature
    possible_keywords = ['public', 'private', 'protected', 'static', 'final', 'abstract']

    # Find the start of the function definition
    while current < len(text):
        # Check if any of the keywords is found
        for keyword in possible_keywords:
            if text[current:].startswith(keyword):
                begin = current
                current += len(keyword)
                break
        else:
            current += 1

    # No function definition found
    if begin == 0:
        raise AssertionError("No function definition found.")

    # Match parenthesis to find the end of the function definition
    paren_count = 0
    while current < len(text):
        if text[current] == '(':
            paren_count += 1
        elif text[current] == ')':
            paren_count -= 1
            if paren_count == 0:
                # Function arguments closed, now look for '{' or the start of exceptions
                brace_or_throws = min(text.find('{', current), text.find('throws', current))
                if brace_or_throws != -1:
                    return text[begin:brace_or_throws].strip()
                else:
                    raise AssertionError("Function definition does not end properly.")
        current += 1

    raise AssertionError("Function definition incomplete.")



def extract_python_function_signature(text):
    """
    Extracts the text before a function body in Python.

    Args:
    text (str): The Python code as a string.

    Returns:
    str: The text before the function body.
    """
    # Store the begin position and current position
    begin = 0
    current = 0

    # Find the 'def' keyword
    def_pos = text.find('def', begin)
    if def_pos == -1:
        raise AssertionError("No function definition found.")

    # Move current position to the end of 'def'
    current = def_pos + len('def')

    # Match parenthesis to find the end of the function definition
    paren_count = 0
    while current < len(text):
        if text[current] == '(':
            paren_count += 1
        elif text[current] == ')':
            paren_count -= 1
            if paren_count == 0:
                # Function arguments closed, now look for ':'
                colon_pos = text.find(':', current)
                if colon_pos != -1:
                    return text[begin:colon_pos]
                else:
                    raise AssertionError("Function definition does not end with a colon.")
        current += 1

    raise AssertionError("Function definition incomplete.")



def extract_signature(function_def, language):
    """
    Extracts the signature of a Python or Java function from its definition,
    including any decorators, annotations, and other preceding content.

    :param function_def: String containing the function definition.
    :param language: The programming language ('python' or 'java').
    :return: The function signature along with the content before it
    """
    if language == 'python':
        return extract_python_function_signature(function_def)
        # Pattern for Python content before and including function signature
        # pattern = r'^(.*?def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)(?:\s*->\s*[\w\[\], ]+)?\s*):'
        # pattern = r'(.*?def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)(?:\s*->\s*[\w\[\], ]+)?)'

    elif language == 'java':
        # Pattern for Java content before and including function signature
        return extract_java_function_signature(function_def)
    else:
        return None


def read_humaneval_data(fname):
    print("Reading Dataset")
    data = read_jsonlines(fname)
    examples = []
    for d in tqdm.tqdm(data):
        ex = {
            "id": d["task_id"],
            "problem": f"Write a function.\nFunction name: {d['entry_point']}\nFunction Description: {d['prompt']}",
            "function_name": d['entry_point'],
            "test_code": d['test']
        }
        examples.append(ex)
    return examples

def read_mbpp_data(fname):
    print("Reading Dataset")
    data = read_jsonlines(fname)
    examples = []
    for idx, d in enumerate(tqdm.tqdm(data)):
        function_signature = ""
        for l in d["code"].splitlines():
            if(l[:3] == "def"):
                function_signature = l
        if(function_signature == ""):
            raise AssertionError("Function Signature Not Found")
        ex = {
            "id": idx,
            "problem": f"Write a function.\nFunction signature: {function_signature}\nFunction Description: {d['text']}",
            "test_code": '\n'.join(d['test_list'])
        }
        examples.append(ex)
    return examples

def read_codereval_data(fname, workdir, args):
    print("Reading Dataset")
    data = read_json(fname)
    file_list = {}
    examples = []
    for d in tqdm.tqdm(data['RECORDS']):
        dir_name = codereval_project_to_directory_name(d['project'])
        if(dir_name not in file_list):
            file_list[dir_name] = list(find_files(os.path.join(workdir, dir_name), ['*.py', '*.java']))
        # Should resolve absolute path of the file, for running the linter
        if("file_path" in d):
            path_kw = d["file_path"]
        elif("file_name" in d):
            path_kw = d["file_name"]
        else:
            raise AssertionError("path is not found")

        path_candidates = []
        for path in file_list[dir_name]:
            if path_kw in path:
                with open(path, 'r') as file:
                    file_content = file.read()
                path_candidates.append((path, similarity(d['file_content'], file_content)))

        if len(path_candidates) == 0:
            raise AssertionError("file is not found")

        # Sort by similarity and select the one with the highest similarity
        path_candidates.sort(key=lambda x: x[1], reverse=True)
        most_similar_file = path_candidates[0][0]
        ex = {
            "id": d["_id"],
            "problem": f"Write a function.\nFunction name: {d['name']}\nFunction signature:  " + extract_signature(d['code'], language=args.split) + f"\nFunction Description:\n{d['human_label']}\n",
            "function_name": d['name'],
            "project": d["project"],
            "lineno": d["lineno"],
            "problem_one-gram": d["problem_one-gram"],
            "problem_ada002": d["problem_ada002"],
            "top_k_context_one-gram": d["top_k_context_one-gram"],
            "top_k_context_ada002": d["top_k_context_ada002"],
            "test_case_file": most_similar_file,
            "file_content": d["file_content"],
            "gold": d['code'],
            "repo_dirname": dir_name,
            "dependency_level": d["level"],
            "oracle_context": d["oracle_context"]
        }
        if("file_path" in d):
            ex["file_path"] = d["file_path"]
        if("file_name" in d):
            ex["file_name"] = d["file_name"]

        examples.append(ex)
    return examples

# hotpot evaluation

