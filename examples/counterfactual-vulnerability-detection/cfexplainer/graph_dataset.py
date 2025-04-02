import sys, json, os
import os.path as osp
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import pickle as pkl
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data, Batch
from tqdm.std import trange
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

from helpers import utils
from helpers import joern
from data_pre import bigvul


class VulGraphDataset(Dataset):
    def __init__(self, root: Optional[str] = "storage/processed/vul_graph_dataset", 
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None, log: bool = True, 
                 encoder = None, tokenizer = None, partition = None,
                 vulonly = False, sample = -1, splits = "default",
                 ):
        os.makedirs(root, exist_ok=True)
        
        self.encoder = encoder
        self.word_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.detach().cpu().numpy() if self.encoder is not None else None
        
        self.tokenizer = tokenizer
        self.partition = partition
        
        self.vulonly = vulonly
        self.sample = sample
        self.splits = splits
        
        super().__init__(root, transform, pre_transform, pre_filter, log)
        
        self.data_list = torch.load(self.processed_paths[0])
        
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'{self.partition}_processed')
    
    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'
    
    def process(self):
        # Get finished samples
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(utils.processed_dir() / "bigvul/before/*nodes*"))
        ]
        self.df = bigvul(splits=self.splits)
        self.df = self.df[self.df.label == self.partition]
        self.df = self.df[self.df.id.isin(self.finished)]

        # Balance set
        vul = self.df[self.df.vul == 1]
        nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
        self.df = pd.concat([vul, nonvul])

        # Small sample (for debugging):
        if self.sample > 0:
            self.df = self.df.sample(self.sample, random_state=0)

        # Filter only vulnerable
        if self.vulonly:
            self.df = self.df[self.df.vul == 1]

        # Filter out samples with no lineNumber from Joern output
        self.df["valid"] = utils.dfmp(
            self.df, VulGraphDataset.check_validity, "id", desc="Validate Samples: "
        )
        self.df = self.df[self.df.valid]

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

        data_list = []
        for idx in trange(self.df.shape[0]):
            _id = self.idx2id[idx]
            n, e = self.feature_extraction(VulGraphDataset.itempath(_id))
            x = np.array(list(n.subseq_feat.values))
            edge_index = np.array(e)
            code_graph = Data(x=torch.FloatTensor(x), edge_index=torch.LongTensor(edge_index))
            
            n["vuln"] = n.id.map(self.get_vuln_indices(_id)).fillna(0)
            code_graph.__setitem__("_VULN", torch.Tensor(n["vuln"].astype(int).to_numpy()))
            code_graph.__setitem__("_LINE", torch.Tensor(n["id"].astype(int).to_numpy()))
            code_graph.__setitem__("_SAMPLE", torch.Tensor([_id] * len(n)))
            data_list.append(code_graph)

        print('Saving...')
        torch.save(data_list, self.processed_paths[0])
        
    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]
    
    def itempath(_id):
        """Get itempath path from item id."""
        return utils.processed_dir() / f"bigvul/before/{_id}.c"
    
    def check_validity(_id):
        """Check whether sample with id=_id has node/edges.

        Example:
        _id = 1320
        with open(str(utils.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
            nodes = json.load(f)
        """
        valid = 0
        try:
            with open(str(VulGraphDataset.itempath(_id)) + ".nodes.json", "r") as f:
                nodes = json.load(f)
                lineNums = set()
                for n in nodes:
                    if "lineNumber" in n.keys():
                        lineNums.add(n["lineNumber"])
                        if len(lineNums) > 1:
                            valid = 1
                            break
                if valid == 0:
                    return False
            with open(str(VulGraphDataset.itempath(_id)) + ".edges.json", "r") as f:
                edges = json.load(f)
                edge_set = set([i[2] for i in edges])
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
                return True
        except Exception as E:
            print(E, str(VulGraphDataset.itempath(_id)))
            return False
        
    def get_vuln_indices(self, _id):
        """Obtain vulnerable lines from sample ID."""
        df = self.df[self.df.id == _id]
        removed = df.removed.item()
        return dict([(i, 1) for i in removed])
    
    def feature_extraction(self, filepath):
        cache_name = "_".join(str(filepath).split("/")[-3:])
        cachefp = utils.get_dir(utils.cache_dir() / "vul_graph_feat") / Path(cache_name).stem
        try:
            with open(cachefp, "rb") as f:
                return pkl.load(f)
        except:
            pass

        try:
            nodes, edges = joern.get_node_edges(filepath)
        except:
            return None
        subseq = (
            nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
            .groupby("lineNumber")
            .head(1)
        )
        subseq = subseq[["lineNumber", "code", "local_type"]].copy()
        subseq.code = subseq.local_type + " " + subseq.code
        subseq = subseq.drop(columns="local_type")
        subseq = subseq[~subseq.eq("").any(axis='columns')]
        subseq = subseq[subseq.code != " "]
        subseq = subseq[subseq.code.notnull()]
        subseq.lineNumber = subseq.lineNumber.astype(int)
        subseq = subseq.sort_values("lineNumber")
        subseq.code = subseq.code.apply(lambda s: ' '.join(s.split()))
        subseq.code = subseq.code.apply(lambda s: [self.tokenizer.cls_token] + self.tokenizer.tokenize(s) + [self.tokenizer.sep_token])
        subseq["code_feat"] = subseq.code.apply(lambda tokens: self.tokenizer.convert_tokens_to_ids(tokens))
        subseq.code = subseq.code.apply(lambda tokens: ' '.join(tokens))
        subseq.code_feat = subseq.code_feat.apply(lambda token_ids: np.mean(self.word_embeddings[token_ids], axis=0))
        subseq_feat = subseq.drop(columns="code")
        subseq = subseq.drop(columns="code_feat")
        subseq = subseq.set_index("lineNumber").to_dict()["code"]
        subseq_feat = subseq_feat.set_index("lineNumber").to_dict()["code_feat"]

        nodesline = nodes[nodes.lineNumber != ""].copy()
        nodesline.lineNumber = nodesline.lineNumber.astype(int)
        nodesline = (
            nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
            .groupby("lineNumber")
            .head(1)
        )
        edgesline = edges.copy()
        edgesline.innode = edgesline.line_in
        edgesline.outnode = edgesline.line_out
        nodesline.id = nodesline.lineNumber
        edgesline = joern.rdg(edgesline, "pdg")
        nodesline = joern.drop_lone_nodes(nodesline, edgesline)
        # Drop duplicate edges
        edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])
        # REACHING DEF to DDG
        edgesline["etype"] = edgesline.apply(
            lambda x: "DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
        )
        edgesline = edgesline[edgesline.innode.apply(lambda x: isinstance(x, float))]
        edgesline = edgesline[edgesline.outnode.apply(lambda x: isinstance(x, float))]
        edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
        edgesline_reverse.columns = ["outnode", "innode", "etype"]
        uedge = pd.concat([edgesline, edgesline_reverse])
        uedge = uedge[uedge.innode != uedge.outnode]
        uedge = uedge.groupby(["innode", "etype"]).agg({"outnode": set})
        uedge = uedge.reset_index()
        if len(uedge) > 0:
            # uedge = uedge.pivot("innode", "etype", "outnode")
            uedge = uedge.pivot(index="innode", columns="etype", values="outnode")
            if "DDG" not in uedge.columns:
                uedge["DDG"] = None
            if "CDG" not in uedge.columns:
                uedge["CDG"] = None
            uedge = uedge.reset_index()[["innode", "CDG", "DDG"]]
            uedge.columns = ["lineNumber", "control", "data"]
            uedge.control = uedge.control.apply(
                lambda x: list(x) if isinstance(x, set) else []
            )
            uedge.data = uedge.data.apply(lambda x: list(x) if isinstance(x, set) else [])
            data = uedge.set_index("lineNumber").to_dict()["data"]
            control = uedge.set_index("lineNumber").to_dict()["control"]
        else:
            data = {}
            control = {}

        # Generate PDG
        pdg_nodes = nodesline.copy()
        pdg_nodes = pdg_nodes[["id"]].sort_values("id")
        pdg_nodes["subseq"] = pdg_nodes.id.map(subseq).fillna("")
        pdg_nodes["subseq_feat"] = pdg_nodes.id.map(subseq_feat).fillna("")
        pdg_nodes["data"] = pdg_nodes.id.map(data)
        pdg_nodes["control"] = pdg_nodes.id.map(control)
        pdg_edges = edgesline.copy()
        pdg_nodes = pdg_nodes.reset_index(drop=True).reset_index()
        pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
        pdg_edges.innode = pdg_edges.innode.map(pdg_dict)
        pdg_edges.outnode = pdg_edges.outnode.map(pdg_dict)
        pdg_edges = pdg_edges.dropna()
        pdg_edges = (pdg_edges.outnode.tolist(), pdg_edges.innode.tolist())

        # Cache
        with open(cachefp, "wb") as f:
            pkl.dump([pdg_nodes, pdg_edges], f)
        return pdg_nodes, pdg_edges


def collate(data_list):
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == '__main__':
    MODEL_CLASSES = {
        'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
        'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
        'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
        't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer)
    }
    
    model_type = "roberta"
    model_name_or_path = "microsoft/graphcodebert-base"
    tokenizer_name = "microsoft/graphcodebert-base"
    
    partition = sys.argv[1]
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

    language_model = model_class.from_pretrained(model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path), config=config)
    
    dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), encoder=language_model, tokenizer=tokenizer, partition=partition)
    print(dataset)
    print(dataset.data_list[0])
    print(dataset.data_list[0].x)
    print(dataset.data_list[0].edge_index)
    print(dataset.data_list[0]._SAMPLE)
