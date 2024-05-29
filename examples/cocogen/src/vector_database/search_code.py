# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import scipy
import tqdm
import os
import copy
import functools

from .utils import Tools, FilePathBuilder, CONSTANTS


class SimilarityScore:
    @staticmethod
    def cosine_similarity(embedding_vec1, embedding_vec2):
        return 1 - scipy.spatial.distance.cosine(embedding_vec1, embedding_vec2)

    @staticmethod
    def jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union


class CodeSearchWorker:
    def __init__(self, repo_embedding_lines, query_embedding_lines, output_path, sim_scorer, max_top_k, log_message):
        self.repo_embedding_lines = repo_embedding_lines  # list
        self.query_embedding_lines = query_embedding_lines  # list
        self.max_top_k = max_top_k
        self.sim_scorer = sim_scorer
        self.output_path = output_path
        self.log_message = log_message

    def _is_context_after_hole(self, repo_embedding_line, query_line):
        hole_fpath_tuple = tuple(query_line['metadata']['fpath_tuple'])
        context_is_not_after_hole = []
        for metadata in repo_embedding_line['metadata']:
            if tuple(metadata['fpath_tuple']) != hole_fpath_tuple:
                context_is_not_after_hole.append(True)
                continue
            # now we know that the repo line is in the same file as the hole
            if metadata['end_line_no'] <= query_line['metadata']['context_start_lineno']:
                context_is_not_after_hole.append(True)
                continue
            context_is_not_after_hole.append(False)
        return not any(context_is_not_after_hole)

    def _find_top_k_context(self, query_line):
        top_k_context = []
        query_embedding = np.array(query_line['data'][0]['embedding'])
        for repo_embedding_line in self.repo_embedding_lines:
            if self._is_context_after_hole(repo_embedding_line, query_line):
                continue
            repo_line_embedding = np.array(repo_embedding_line['data'][0]['embedding'])
            similarity_score = self.sim_scorer(query_embedding, repo_line_embedding)
            top_k_context.append((repo_embedding_line, similarity_score))
        top_k_context = sorted(top_k_context, key=lambda x: x[1], reverse=False)[-self.max_top_k:]
        return top_k_context

    def run(self):
        query_lines_with_retrieved_results = []
        for query_line in self.query_embedding_lines:
            new_line = copy.deepcopy(query_line)
            top_k_context = self._find_top_k_context(new_line)
            new_line['top_k_context'] = top_k_context
            query_lines_with_retrieved_results.append(new_line)
        Tools.dump_pickle(query_lines_with_retrieved_results, self.output_path)


class CodeSearchWorker_SingleQuery:
    def __init__(self, repo_embedding_lines, query_data, query_embedding_vector, sim_scorer, max_top_k, log_message,
                 sim_method):
        self.repo_embedding_lines = repo_embedding_lines  # list
        self.query_data = query_data
        self.query_embedding_vector = query_embedding_vector
        self.max_top_k = max_top_k
        self.sim_scorer = sim_scorer
        # self.output_path = output_path
        self.log_message = log_message
        self.similarity_method = sim_method

    def _is_context_after_hole(self, repo_embedding_line, query_line):
        try:
            hole_fpath_str = query_line['file_path']
        except:
            hole_fpath_str = query_line['file_name']

        # hole_fpath_tuple = tuple(query_line['metadata']['fpath_tuple'])
        context_is_not_after_hole = []
        for metadata in repo_embedding_line['metadata']:
            repo_fpath_str = '/'.join(metadata['fpath_tuple'])
            if (hole_fpath_str not in repo_fpath_str):
                context_is_not_after_hole.append(True)
                continue
            # if tuple(metadata['fpath_tuple']) != hole_fpath_tuple:
            #     context_is_not_after_hole.append(True)
            #     continue
            # now we know that the repo line is in the same file as the hole
            if metadata['end_line_no'] <= int(query_line['lineno']):
                context_is_not_after_hole.append(True)
                continue
            # if metadata['end_line_no'] <= query_line['metadata']['context_start_lineno']:
            #     context_is_not_after_hole.append(True)
            #     continue
            context_is_not_after_hole.append(False)
        return not any(context_is_not_after_hole)

    def _find_top_k_context(self, query_line, query_embedding):
        top_k_context = []
        # query_embedding = np.array(query_line['data'][0]['embedding'])
        for repo_embedding_line in self.repo_embedding_lines:
            if self._is_context_after_hole(repo_embedding_line, query_line):
                continue
            repo_line_embedding = np.array(repo_embedding_line['data'][0]['embedding'])
            similarity_score = self.sim_scorer(query_embedding, repo_line_embedding)
            top_k_context.append((repo_embedding_line, similarity_score))
        top_k_context = sorted(top_k_context, key=lambda x: x[1], reverse=False)[-self.max_top_k:]
        return top_k_context

    def run(self):
        # query_lines_with_retrieved_results = []
        query_embedding = copy.deepcopy(self.query_embedding_vector)
        top_k_context = self._find_top_k_context(self.query_data, query_embedding)
        self.query_data['top_k_context' + "_" + self.similarity_method] = top_k_context
        # query_lines_with_retrieved_results.append(new_line)
        # Tools.dump_pickle(query_lines_with_retrieved_results, self.output_path)


class CodeSearchWrapper:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        if vectorizer == 'one-gram':
            self.sim_scorer = SimilarityScore.jaccard_similarity
            self.vector_path_builder = FilePathBuilder.one_gram_vector_path
            self.field_name = "problem_one-gram"
        elif vectorizer == 'ada002':
            self.sim_scorer = SimilarityScore.cosine_similarity
            self.vector_path_builder = FilePathBuilder.ada002_vector_path
            self.field_name = "problem_ada002"
        self.max_top_k = 20  # store 20 top k context for the prompt construction (top 10)

    def search_codereval(self, data, repo_name, directory_name, window_size, slice_size):
        # 1. open corresponding database

        repo_window_path = FilePathBuilder.repo_windows_path(directory_name, window_size, slice_size)  # Good
        repo_embedding_path = self.vector_path_builder(repo_window_path)  # Good

        # use single-thread
        repo_embedding_lines = Tools.load_pickle(repo_embedding_path)
        log_message = f'repo: {repo_name}, window: {window_size}, slice: {slice_size}  {self.vectorizer}, max_top_k: {self.max_top_k}'
        worker = CodeSearchWorker_SingleQuery(repo_embedding_lines=repo_embedding_lines, query_data=data,
                                              query_embedding_vector=data[self.field_name],
                                              sim_scorer=self.sim_scorer,
                                              max_top_k=self.max_top_k,
                                              log_message=log_message,
                                              sim_method=self.vectorizer)
        worker.run()
        pass
        # process pool
        # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        #     futures = {executor.submit(worker.run, ) for worker in workers}
        #     for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        #         future.result()
        #
        # pass


class DependencySearchWorker:
    def __init__(self, repo_embedding_lines, query_embedding_vector, query_symbols, query_symbol_field,
                 search_scope, scope_field,
                 sim_scorer,
                 max_top_k, log_message,
                 sim_method):
        self.repo_embedding_lines = repo_embedding_lines  # list
        self.query_embedding_vector = query_embedding_vector
        self.max_top_k = max_top_k
        self.sim_scorer = sim_scorer
        self.query_symbols = query_symbols
        self.query_symbol_field = query_symbol_field
        self.log_message = log_message
        self.similarity_method = sim_method
        self.search_scope = search_scope
        self.scope_field = scope_field

    def _find_top_k_context(self, query_embedding):
        top_k_context = []
        exact_match_count = 0
        exact_match_limit = 2
        # query_embedding = np.array(query_line['data'][0]['embedding'])
        for repo_embedding_line in self.repo_embedding_lines:
            if(self.search_scope is not None):
                if (repo_embedding_line['metadata'][self.scope_field].split(' ')[1] != self.search_scope):
                    continue  # skip not-in-scope stuff
            # check symbol name match
            if(self.query_symbols is not None):
                for query_symbol in self.query_symbols:
                    # check scope
                    if (repo_embedding_line['metadata'][self.query_symbol_field] == query_symbol
                            and exact_match_count < exact_match_limit):
                        top_k_context.append((repo_embedding_line, 1.0))  # exact match
                        exact_match_count += 1
                        continue

            repo_line_embedding = np.array(repo_embedding_line['data'][0]['embedding'])
            similarity_score = self.sim_scorer(query_embedding, repo_line_embedding)
            top_k_context.append((repo_embedding_line, similarity_score))
        top_k_context = sorted(top_k_context, key=lambda x: x[1], reverse=False)[-self.max_top_k:]
        return top_k_context

    def run(self):
        # query_lines_with_retrieved_results = []
        query_embedding = copy.deepcopy(self.query_embedding_vector)
        top_k_context = self._find_top_k_context(query_embedding)
        return top_k_context
        # self.query_data['top_k_context' + "_" + self.similarity_method] = top_k_context
        # query_lines_with_retrieved_results.append(new_line)
        # Tools.dump_pickle(query_lines_with_retrieved_results, self.output_path)


class DependencyTableSearchWrapper:
    def __init__(self):
        self.vectorizer = 'ada002'
        self.sim_scorer = SimilarityScore.cosine_similarity
        self.vector_path_builder = FilePathBuilder.ada002_dependency_table_path
        self.max_top_k = 10  # store 20 top k context for the prompt construction (top 10)

    def search_dependency_table(self, repo_name, table_name, embedding_vector, symbol_names=None, symbol_field=None, search_scope=None, scope_field=None):
        # 1. open corresponding database

        table_path = FilePathBuilder.search_table_dependency_path(repo=repo_name, table_name=table_name)  # Good
        repo_embedding_path = self.vector_path_builder(table_path)  # Good

        # use single-thread
        repo_embedding_lines = Tools.load_pickle(repo_embedding_path)
        log_message = f'repo: {repo_name}, max_top_k: {self.max_top_k}'
        worker = DependencySearchWorker(repo_embedding_lines=repo_embedding_lines,
                                        query_embedding_vector=embedding_vector,
                                        query_symbols=symbol_names,
                                        query_symbol_field=symbol_field,
                                        search_scope=search_scope,
                                        scope_field=scope_field,
                                        sim_scorer=self.sim_scorer,
                                        max_top_k=self.max_top_k,
                                        log_message=log_message,
                                        sim_method=self.vectorizer)
        return worker.run()
