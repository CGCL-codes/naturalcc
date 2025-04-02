import numpy as np
import scipy
import tqdm
import os
import copy
from vector_database.utils import Tools, FilePathBuilder, CONSTANTS


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


class RepoCoderVectorQueryWorker:
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
            # now we know that the repo line is in the same file as the hole
            if metadata['end_line_no'] <= int(query_line['lineno']):
                context_is_not_after_hole.append(True)
                continue
            context_is_not_after_hole.append(False)
        return not any(context_is_not_after_hole)

    def _find_top_k_context(self, query_line, query_embedding):
        top_k_context = []
        for repo_embedding_line in self.repo_embedding_lines:
            if self._is_context_after_hole(repo_embedding_line, query_line):
                continue
            repo_line_embedding = np.array(repo_embedding_line['data'][0]['embedding'])
            similarity_score = self.sim_scorer(query_embedding, repo_line_embedding)
            top_k_context.append((repo_embedding_line, similarity_score))
        top_k_context = sorted(top_k_context, key=lambda x: x[1], reverse=False)[-self.max_top_k:]
        return top_k_context

    def search(self):
        # query_lines_with_retrieved_results = []
        query_embedding = copy.deepcopy(self.query_embedding_vector)
        top_k_context = self._find_top_k_context(self.query_data, query_embedding)
        self.query_data['top_k_context' + "_" + self.similarity_method] = top_k_context
        return top_k_context


class StructuralSymbolQueryWorker:

    def __init__(self, repo_embedding_lines, query_data, query_symbol_name):
        self.repo_embedding_lines = repo_embedding_lines  # list
        self.query_data = query_data
        self.query_embedding_vector = query_symbol_name

    def _is_context_after_hole(self, repo_embedding_line, query_line):
        try:
            hole_fpath_str = query_line['file_path']
        except:
            hole_fpath_str = query_line['file_name']

        context_is_not_after_hole = []
        repo_fpath_str = repo_embedding_line['metadata']['definition_document']
        if (hole_fpath_str not in repo_fpath_str):
            context_is_not_after_hole.append(True)
        elif int(repo_embedding_line['metadata']['end_line_no']) <= int(query_line['lineno']):
            context_is_not_after_hole.append(True)
        else:
            pass
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

    def search(self):
        query_embedding = copy.deepcopy(self.query_embedding_vector)
        top_k_context = self._find_top_k_context(self.query_data, query_embedding)
        return top_k_context


class RepoCoderCodeSearchWrapper:
    repo_name_to_directory = {
        "ansible-security/ansible_collections.ibm.qradar": "ansible-security---ansible_collections.ibm.qradar",
        "champax/pysolbase": "champax---pysolbase",
        "gopad/gopad-python": "gopad---gopad-python",
        "mozilla/relman-auto-nag": "mozilla---relman-auto-nag",
        "openstack/neutron-lib": "openstack---neutron-lib",
        "pre-commit/pre-commit": "pre-commit---pre-commit",
        "scieloorg/packtools": "scieloorg---packtools",
        "SoftwareHeritage/swh-lister": "SoftwareHeritage---swh-lister",
        "witten/atticmatic": "witten---atticmatic",
        "awsteiner/o2sclpy": "awsteiner---o2sclpy",
        "cloudmesh/cloudmesh-common": "cloudmesh---cloudmesh-common",
        "ikus060/rdiffweb": "ikus060---rdiffweb",
        "MozillaSecurity/lithium": "MozillaSecurity---lithium",
        "ossobv/planb": "ossobv---planb",
        "rak-n-rok/Krake": "rak-n-rok---Krake",
        "scrolltech/apphelpers": "scrolltech---apphelpers",
        "standalone": "standalone",
        "witten/borgmatic": "witten---borgmatic",
        "bastikr/boolean": "bastikr---boolean",
        "commandline/flashbake": "commandline---flashbake",
        "infobloxopen/infoblox-client": "infobloxopen---infoblox-client",
        "mwatts15/rdflib": "mwatts15---rdflib",
        "pexip/os-python-cachetools": "pexip---os-python-cachetools",
        "redhat-openstack/infrared": "redhat-openstack---infrared",
        "SEED-platform/py-seed": "SEED-platform---py-seed",
        "sunpy/radiospectra": "sunpy---radiospectra",
        "ynikitenko/lena": "ynikitenko---lena",
        "bazaar-projects/docopt-ng": "bazaar-projects---docopt-ng",
        "cpburnz/python-sql-parameters": "cpburnz---python-sql-parameters",
        "jaywink/federation": "jaywink---federation",
        "neo4j/neo4j-python-driver": "neo4j---neo4j-python-driver",
        "pexip/os-python-dateutil": "pexip---os-python-dateutil",
        "rougier/matplotlib": "rougier---matplotlib",
        "sipwise/repoapi": "sipwise---repoapi",
        "turicas/rows": "turicas---rows",
        "zimeon/ocfl-py": "zimeon---ocfl-py",
        "burgerbecky/makeprojects": "burgerbecky---makeprojects",
        "eykd/prestoplot": "eykd---prestoplot",
        "kirankotari/shconfparser": "kirankotari---shconfparser",
        "openstack/cinder": "openstack---cinder",
        "pexip/os-zope": "pexip---os-zope",
        "santoshphilip/eppy": "santoshphilip---eppy",
        "skorokithakis/shortuuid": "skorokithakis---shortuuid",
        "ufo-kit/concert": "ufo-kit---concert",
        "atmosphere-atmosphere-2.7.x": "atmosphere",
        "fastjson2-main": "fastjson2",
        "framework-master": "framework",
        "hasor-master": "hasor",
        "jgrapht-master": "jgrapht",
        "jjwt-master": "jjwt",
        "logging-log4j1-main": "logging-log4j1",
        "protostuff-master": "protostuff",
        "skywalking-master": "skywalking",
        "interviews-master": "interviews"
    }

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

    def search(self, data):
        # 1. open corresponding database
        repo_name = data['project']
        directory_name = self.repo_name_to_directory[repo_name]
        window_size = 20
        slice_size = 2
        repo_window_path = FilePathBuilder.repo_windows_path(directory_name, window_size, slice_size)  # Good
        repo_embedding_path = self.vector_path_builder(repo_window_path)  # Good
        repo_embedding_path = os.path.join('vector_database', self.vector_path_builder(repo_window_path))  # Good

        # use single-thread
        repo_embedding_lines = Tools.load_pickle(repo_embedding_path)
        log_message = f'repo: {repo_name}, window: {window_size}, slice: {slice_size}  {self.vectorizer}, max_top_k: {self.max_top_k}'
        worker = RepoCoderVectorQueryWorker(repo_embedding_lines=repo_embedding_lines, query_data=data,
                                            query_embedding_vector=data[self.field_name],
                                            sim_scorer=self.sim_scorer,
                                            max_top_k=self.max_top_k,
                                            log_message=log_message,
                                            sim_method=self.vectorizer)
        return worker.search()
