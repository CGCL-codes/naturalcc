import torch
from rank_bm25 import BM25Okapi
from typing import List
from multiprocessing import Pool, cpu_count
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import tokenize_nltk
from transformers import AutoModel, AutoTokenizer, AutoConfig
from Levenshtein import distance as levenshtein_distance
import warnings
# 新增代码
CODE_SAGE_MODELS = {
    "codebert": "/mnt/silver/tanlei/hf-models/codebert-base",
    "unixcoder": "/mnt/silver/tanlei/hf-models/unixcoder-base"
}

def edit_similarity(s1, s2):
    """基于python-Levenshtein的标准化编辑相似度"""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0  # 空字符串与空字符串的相似度为1
    edit_dist = levenshtein_distance(s1, s2)
    return 1 - (edit_dist / max_len)

def jaccard_similarity(tokenized_query, tokenized_doc, containment=False):
    set1 = set(tokenized_query)
    set2 = set(tokenized_doc)
    intersection = len(set1.intersection(set2))
    union = len(set1) if containment else len(set1.union(set2))
    return float(intersection) / union


def tokenize_corpus(corpus, tokenizer_fn):
    pool = Pool(cpu_count())
    tokenized_corpus = pool.map(tokenizer_fn, corpus)
    return tokenized_corpus


def tokenize_query_and_docs(query, docs):
    tokenized_query = tokenize_nltk(query)
    tokenized_docs = [tokenize_nltk(d) for d in docs]
    return tokenized_query, tokenized_docs

def min_max_normalization(scores):
    if len(scores) == 0:
        return scores
    min_val = min(scores)
    max_val = max(scores)
    denominator = max_val - min_val
    if denominator == 0:
        return [1.0] * len(scores)  # 避免除以零的情况
    return [(s - min_val) / (denominator + 1e-8) for s in scores]


def lexical_ranking(
        query,
        docs,
        ranking_fn,
        doc_ids=None,
        score_threshold=None,
):
    if ranking_fn == "bm25":
        tokenized_query, tokenized_docs = tokenize_query_and_docs(query, docs)
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)
        scores = min_max_normalization(scores)
    elif ranking_fn == "tfidf":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'")
            tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_nltk)
            X = tfidf_vectorizer.fit_transform(docs).toarray()  # (n_fn, n_features)
            y = tfidf_vectorizer.transform([query]).toarray()  # (1, n_features)
            scores = cosine_similarity(X, y).tolist()  # (n_fn, 1)
    elif ranking_fn == "jaccard_sim":
        tokenized_query, tokenized_docs = tokenize_query_and_docs(query, docs)
        scores = [jaccard_similarity(tokenized_query, d, containment=False) for d in tokenized_docs]
    elif ranking_fn == "edit_distance":
        tokenized_query, tokenized_docs = tokenize_query_and_docs(query, docs)
        scores = [
            edit_similarity(' '.join(tokenized_query), ' '.join(d)) 
            for d in tokenized_docs
        ]
    else:
        raise NotImplementedError

    if score_threshold:
        skip_ids = [idx for idx, s in enumerate(scores) if s < score_threshold]
        scores = [s for idx, s in enumerate(scores) if idx not in skip_ids]
        docs = [d for idx, d in enumerate(docs) if idx not in skip_ids]
        if doc_ids is not None:
            doc_ids = [doc_id for idx, doc_id in enumerate(doc_ids) if idx not in skip_ids]

    if len(docs) == 0:
        return docs, doc_ids, scores

    if doc_ids is not None:
        doc_ids = [x for _, x in sorted(zip(scores, doc_ids), reverse=True)]
    docs_scores = [(x, s) for s, x in sorted(zip(scores, docs), reverse=True)]
    docs = [item[0] for item in docs_scores]
    scores = [item[1] for item in docs_scores]

    return docs, doc_ids, scores


class SemanticReranking:

    def __init__(self, model_type="unixcoder", **kwargs):
        self.model_type = model_type
        model_name = CODE_SAGE_MODELS[model_type]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        
        # CodeBERT需要特殊处理attention_mask
        if model_type == "codebert":
            config.output_hidden_states = True  # 启用隐藏层输出
            
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.max_sequence_length = kwargs.get("max_sequence_length", 256)
        
    def text_to_tensor(
            self,
            text: str,
            pad_to_max: bool = True,
    ):
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.max_sequence_length,
            pad_to_max_length=False,
            truncation=True
        )

        if pad_to_max and len(token_ids) < self.max_sequence_length:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(token_ids))
        if len(token_ids) > self.max_sequence_length:
            token_ids = token_ids[0:self.max_sequence_length]

        return torch.tensor(token_ids)

    def get_pad_id(self):
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor):
        return tokens_tensor != self.get_pad_id()

    def get_representations(self, list_input_ids, gpu_id):
        device = torch.device('cuda', gpu_id)
        self.model = self.model.to(device=device, dtype=torch.float16)
        self.model.eval()

        batch_size = 16
        sequence_outputs = []
        pooled_outputs = []

        for idx in range(0, len(list_input_ids), batch_size):
            start, end = idx, min(idx + batch_size, len(list_input_ids))
            input_ids = torch.stack(list_input_ids[start:end], dim=0).to(device=device)
            attention_mask = self.get_attn_mask(input_ids)

            if self.model_type in CODE_SAGE_MODELS.keys():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                token_embeddings = output.hidden_states[-1]  # bsz x seq_len x hid_dim
            else:
                output = self.model(input_ids, attention_mask)
                token_embeddings = output.last_hidden_state  # bsz x seq_len x hid_dim

            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            sequence_embeddings = sum_embeddings / sum_mask  # bsz x hid_dim

            sequence_outputs.append(token_embeddings)
            pooled_outputs.append(sequence_embeddings)

        sequence_output = torch.cat(sequence_outputs)
        pooled_output = torch.cat(pooled_outputs)

        return sequence_output, pooled_output

    def rerank(self, query: str, docs: List[str], doc_ids: List[str] = None, gpu_id=0, score_threshold=None):
        with torch.no_grad():
            batch_queries = [self.text_to_tensor(query)]
            batch_candidates = [self.text_to_tensor(d) for d in docs]

            _, query_rep = self.get_representations(batch_queries, gpu_id)  # 1 x hidden_size
            _, candi_rep = self.get_representations(batch_candidates, gpu_id)  # num_cand x hidden_size
            scores = torch.nn.functional.cosine_similarity(query_rep, candi_rep).tolist()  # num_cand

        if score_threshold:
            skip_ids = [idx for idx, s in enumerate(scores) if s < score_threshold]
            scores = [s for idx, s in enumerate(scores) if idx not in skip_ids]
            docs = [d for idx, d in enumerate(docs) if idx not in skip_ids]
            if doc_ids is not None:
                doc_ids = [doc_id for idx, doc_id in enumerate(doc_ids) if idx not in skip_ids]

        if len(docs) == 0:
            return docs, doc_ids, scores

        if doc_ids is not None:
            doc_ids = [x for _, x in sorted(zip(scores, doc_ids), reverse=True)]
        docs_scores = [(x, s) for s, x in sorted(zip(scores, docs), reverse=True)]
        docs = [item[0] for item in docs_scores]
        scores = [item[1] for item in docs_scores]

        return docs, doc_ids, scores
