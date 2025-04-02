from ncc2.tasks import NccTask
from ncc2.models.sequence import SequenceBatch
from ncc2.nn.padding import PaddingMask
import torch.nn.functional as F
import torch
from ncc2.data.dataset import NccDataset
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
from ncc2.tasks import register_task
from ncc2.tasks.utils.convertor_registry import ConvertorRegistry
from ncc2.tasks.utils.task_registry import TaskRegistry
from dataclasses import dataclass
from ncc2.models.utils.arch_registry import ArchitectureRegistry
from ncc2.models.llama import LLaMABuilder,LLaMAConfig,LLaMATokenizer,LLaMAConfig,llama_archs
from ncc2.data.text import TextTokenizer

@dataclass
class RetrievalGenerationConfig:
    convertor: dict
    archs: ArchitectureRegistry
    model_name: str
    builder: object
    tokenizer_cls: TextTokenizer


@generation_task('codellama_7b_retrieval_generation')
def codellama() -> RetrievalGenerationConfig:
    key_map = {}
    key_map['tok_embeddings.weight'] = 'decoder_frontend.embed.weight'
    key_map['norm.weight'] = 'decoder.layer_norm.weight'
    key_map['output.weight'] = 'final_proj.weight'
    for i in range(32):
        key_map[f'layers.{i}.attention_norm.weight'] = f'decoder.layers.{i}.self_attn_layer_norm.weight'
        key_map[f'layers.{i}.attention.wq.weight'] = f'decoder.layers.{i}.self_attn.q_proj.weight'
        key_map[f'layers.{i}.attention.wk.weight'] = f'decoder.layers.{i}.self_attn.k_proj.weight'
        key_map[f'layers.{i}.attention.wv.weight'] = f'decoder.layers.{i}.self_attn.v_proj.weight'
        key_map[f'layers.{i}.attention.wo.weight'] = f'decoder.layers.{i}.self_attn.output_proj.weight'
        key_map[f'layers.{i}.feed_forward.w1.weight'] = f'decoder.layers.{i}.ffn.gate_proj.weight'
        key_map[f'layers.{i}.feed_forward.w2.weight'] = f'decoder.layers.{i}.ffn.output_proj.weight'
        key_map[f'layers.{i}.feed_forward.w3.weight'] = f'decoder.layers.{i}.ffn.inner_proj.weight'
        key_map[f'layers.{i}.ffn_norm.weight'] = f'decoder.layers.{i}.ffn_layer_norm.weight'
    return RetrievalGenerationConfig(
        convertor=key_map,
        archs=llama_archs,
        model_name='7b_code',
        builder=LLaMABuilder,
        tokenizer_cls=LLaMATokenizer
    )