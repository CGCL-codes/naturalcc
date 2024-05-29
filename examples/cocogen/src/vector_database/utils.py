# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import glob
import pickle
import json
import time
import re
import tiktoken
from transformers import AutoTokenizer

from openai import OpenAI
client = OpenAI(api_key="xxxxxx")
class CONSTANTS:
    # regular version for Codex
    api_benchmark = 'random_api'
    line_benchmark = 'random_line'
    # short version for CodeGen
    short_api_benchmark = 'short_api'
    short_line_benchmark = 'short_line'
    gt = 'gt'
    rg = 'r-g' # vanilla retrieval-augmented approach
    rgrg = 'r-g-r-g' # RepoCoder, two-stage retrieval and generation

class FilePathBuilder:
    api_completion_benchmark = 'datasets/random-api-completion.test.jsonl'
    random_line_completion_benchmark = 'datasets/random-line-completion.test.jsonl'
    # short version for codegen
    short_api_completion_benchmark = 'datasets/random-api-completion-short-version.test.jsonl'
    short_random_line_completion_benchmark = 'datasets/random-line-completion-short-version.test.jsonl'
    repo_base_dir = '../../../8_codereval/CoderEval_ds/java'

    @staticmethod
    def make_needed_dir(file_path):
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def repo_windows_path(repo, window_size, slice_size):
        out_path = os.path.join('cache/window/repos', f'{repo}_ws{window_size}_slice{slice_size}.pkl')
        FilePathBuilder.make_needed_dir(out_path)
        return out_path

    @staticmethod
    def repo_dependency_path(repo, table_name):
        out_path = os.path.join('../data/CoderEval_symbol_tables', repo, f'{table_name}.json')
        # FilePathBuilder.make_needed_dir(out_path)
        return out_path

    @staticmethod
    def search_table_dependency_path(repo, table_name):
        out_path = os.path.join('data/CoderEval_symbol_tables', repo, f'{table_name}.json')
        # FilePathBuilder.make_needed_dir(out_path)
        return out_path



    @staticmethod
    def one_gram_vector_path(window_file):
        vector_path = window_file.replace('/window/', '/vector/')
        out_path = vector_path.replace('.pkl', '.one-gram.pkl')
        FilePathBuilder.make_needed_dir(out_path)
        return out_path


    @staticmethod
    def ada002_vector_path(window_file):
        vector_path = window_file.replace('/window/', '/vector/')
        out_path = vector_path.replace('.pkl', '.ada002.pkl')
        FilePathBuilder.make_needed_dir(out_path)
        return out_path

    @staticmethod
    def one_gram_ctx_graph_path(window_file):
        vector_path = window_file.replace('/window/', '/vector/')
        out_path = vector_path.replace('.pkl', '.one-gram-ctx-graph.pkl')
        FilePathBuilder.make_needed_dir(out_path)
        return out_path

    @staticmethod
    def ada002_ctx_grpah_path(window_file):
        vector_path = window_file.replace('/window/', '/vector/')
        out_path = vector_path.replace('.pkl', '.ada002-ctx-graph.pkl')
        FilePathBuilder.make_needed_dir(out_path)
        return out_path

    @staticmethod
    def ada002_dependency_table_path(table_file):
        vector_path = table_file.replace('.json', '.ada002-dep-table.pkl')
        out_path = vector_path
        FilePathBuilder.make_needed_dir(out_path)
        return out_path

    @staticmethod
    def retrieval_results_path(query_vector_file, repo_vector_file, max_top_k):
        retrieval_base_dir = os.path.dirname(query_vector_file.replace('/vector/', '/retrieval/'))
        query_file_name = os.path.basename(query_vector_file)
        if query_file_name.endswith('.one-gram.pkl'):
            query_file_name = query_file_name[:-len('.one-gram.pkl')]
        elif query_file_name.endswith('.ada002.pkl'):
            query_file_name = query_file_name[:-len('.ada002.pkl')]
        repo_file_name = os.path.basename(repo_vector_file)[:-len('.pkl')]
        out_path = os.path.join(retrieval_base_dir, f'{query_file_name}.{repo_file_name}.top{max_top_k}.pkl')
        FilePathBuilder.make_needed_dir(out_path)
        return out_path


class CodexTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("p50k_base")
    
    def tokenize(self, text):
        # return self.tokenizer.encode(text)
        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


class Tools:
    @staticmethod
    def read_code(fname):
        with open(fname, 'r', encoding='utf8') as f:
            return f.read()
    
    @staticmethod
    def load_pickle(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def dump_pickle(obj, fname):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def dump_json(obj, fname):
        with open(fname, 'w', encoding='utf8') as f:
            json.dump(obj, f)

    @staticmethod
    def dump_jsonl(obj, fname):
        with open(fname, 'w', encoding='utf8') as f:
            for item in obj:
                f.write(json.dumps(item) + '\n')
    
    @staticmethod
    def load_jsonl(fname):
        with open(fname, 'r', encoding='utf8') as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            return lines
    
    @staticmethod
    def iterate_repository(repo):
        base_dir = FilePathBuilder.repo_base_dir

        # Exclude test cases for Python files
        py_pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.py")
        py_files = glob.glob(py_pattern, recursive=True)
        exclude_py_pattern = r".*_passk_validte[0-9]?\.py$"
        py_files = [f for f in py_files if not re.match(exclude_py_pattern, f)]

        # Exclude test cases for Java files
        java_pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.java")
        java_files = glob.glob(java_pattern, recursive=True)
        exclude_java_pattern = r".*_.*\.java$"
        java_files = [f for f in java_files if not re.match(exclude_java_pattern, f.split('/')[-1])]

        # Combine the lists
        files = py_files + java_files


        #
        # # For Python Files, do these
        # pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.py")
        # files = glob.glob(pattern, recursive=True)
        #
        #
        # # Define your filter pattern or condition here
        # exclude_pattern = r".*_passk_validte[0-9]?\.py$"
        #
        # # Filter the files list using the defined pattern or condition
        # files = [f for f in files if not re.match(exclude_pattern, f)]
        #
        # # For Java files, do these:
        # pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.java")
        # files = glob.glob(pattern, recursive=True)
        #
        #
        # # Define your filter pattern or condition here
        # exclude_pattern = r".*_.*\.java$"
        #
        # # Filter the files list using the defined pattern or condition
        # files = [f for f in files if not re.match(exclude_pattern, f)]
        #
        # # combine the two lists and get the final 'files'
        # files = ...

        skipped_files = []
        loaded_code_files = dict()
        base_dir_list = os.path.normpath(base_dir).split(os.sep)
        for fname in files:
            try:
                code = Tools.read_code(fname)
                fpath_tuple = tuple(os.path.normpath(fname).split(os.sep)[len(base_dir_list):])
                loaded_code_files[fpath_tuple]= code
            except Exception as e:
                skipped_files.append((fname, e))
                continue

        if len(skipped_files) > 0:
            print(f"Skipped {len(skipped_files)} out of {len(files)} files due to I/O errors")
            for fname, e in skipped_files:
                print(f"{fname}: {e}")
        return loaded_code_files

    @staticmethod
    def tokenize(code):
        tokenizer = CodexTokenizer()
        tokenized = tokenizer.tokenize(code)
        return tokenized

    @staticmethod
    def ada002_embedding(code):
        if(code == ""):
            code = " "
        attempts = 5
        for attempt in range(attempts):
            try:
                # text = code.replace("\n", " ")  # Uncomment this line if you want to preprocess the code text
                embed = client.embeddings.create(input=[code], model="text-embedding-ada-002").data[0].embedding
                return embed
            except Exception as e:
                if attempt < attempts - 1:  # If not the last attempt, sleep and then retry
                    time.sleep(5)
                else:
                    print(f"Failed generate embedding for code: [{code}]")
                    return [1.0]*1536
                    #raise e  # Raise the last exception after all attempts fail

        raise Exception("Failed to create embedding after multiple attempts")



