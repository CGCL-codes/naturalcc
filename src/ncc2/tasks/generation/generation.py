from ncc2.tasks import NccTask
from ncc2.models.sequence import SequenceBatch
from ncc2.nn.padding import PaddingMask
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset,DataLoader
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


class NccDataset(Dataset):
    def __init__(self,input):
        self.input = input
        
    def __len__(self):
        return len(self.input)
        
    def __getitem__(self,index):
        if index >= len(self.input):
            raise IndexError('Index out of range')
        sample = {
            'input': self.input[index]
        }
        return sample

@dataclass
class GenerationTaskConfig:
    convertor: dict
    archs: ArchitectureRegistry
    model_name: str
    builder: object
    tokenizer_cls: TextTokenizer     
    
generation_tasks = TaskRegistry('generation')
generation_task = generation_tasks.marker

@generation_task('codellama_7b_code')
def codellama() -> GenerationTaskConfig:
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
    return GenerationTaskConfig(
        convertor=key_map,
        archs=llama_archs,
        model_name='7b_code',
        builder=LLaMABuilder,
        tokenizer_cls=LLaMATokenizer
    )
    
@register_task('generation')
class GenerationTask(NccTask):
    @classmethod
    def __init__(self,config: GenerationTaskConfig=None,task_name: str=None,*args,**kwargs):
        if not config:
            if task_name:
                config = generation_tasks.get_config(task_name)
            else:
                raise ValueError('Config or task_name are needed')
        super().__init__(*args,**vars(config),**kwargs)
              
    @classmethod
    def generate(self,input,max_length=100,top_k=10,top_p=0.95,temperature=0.2,penalty_weight=0.5,penalty_decay=0.95,seed=None,bos_token_id=1,eos_token_id=2,unkown_token_id=0,tokenizer=None):
        if not tokenizer:
            if self.tokenizer:
                tokenizer = self.tokenizer
            else:
                raise Exception('No tokenizer loaded')
        if seed:
            torch.manual_seed(2618)
        encoder = tokenizer.create_encoder()
        decoder = tokenizer.create_decoder()
        
        if isinstance(input,str):
            sqs = [input]
        elif isinstance(input,list):
            sqs = input
        else:
            raise ValueError('Input should be str|list[str]')
        
        penalty = torch.stack([torch.zeros(tokenizer.vocab_info.size) for _ in sqs]).to(self.device)
        sqs_token_ids = list(map(lambda x:encoder(x)[:-1].tolist(),sqs))
        # 初始化重复惩罚
        for i,p in enumerate(penalty):
            p *= penalty_decay
            sq_token_ids = sqs_token_ids[i]
            for token_id in sq_token_ids:
                if not token_id in [bos_token_id,eos_token_id,unkown_token_id]:
                    p[token_id] += penalty_weight
        # 长度对齐，左填充padding      
            sqs_max_length = max([len(x) for x in sqs_token_ids])  
            if sqs_max_length >= max_length:
                break    
            for i,st in enumerate(sqs_token_ids):
                if len(st) < sqs_max_length:
                    sqs_token_ids[i] = [unkown_token_id]*(sqs_max_length-len(st)) + st
        sqs_token_ids = torch.tensor(sqs_token_ids,dtype=torch.int64).to(self.device)                    
        while True:
            # 长度对齐，左填充padding      
            sqs_max_length = max([len(x) for x in sqs_token_ids])  
            if sqs_max_length >= max_length:
                break    
            for i,st in enumerate(sqs_token_ids):
                if len(st) < sqs_max_length:
                    sqs_token_ids[i] = [unkown_token_id]*(sqs_max_length-len(st)) + st
                sqs_token_ids[i] = st[:max_length]
            # 添加mask
            mask = PaddingMask(torch.tensor([sqs_max_length]*len(sqs_token_ids)).to(self.device),sqs_max_length)
            batch = SequenceBatch(sqs_token_ids,mask)
            # forward
            result = self.model.forward(batch)
            last_token_logits = result.logits[:, -1, :]
            # 减去惩罚惩罚
            last_token_logits -= penalty
            last_token_logits[last_token_logits<0] = 0
            last_token_logits = F.softmax(last_token_logits / temperature, dim=-1)
            topk_values, topk_indices = torch.topk(last_token_logits,top_k)
            sorted_probs, sorted_indices = torch.sort(topk_values, descending=True) 
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            selected_indices = [(row[cumulative_probs[i] <= top_p] if cumulative_probs[i][0]<=top_p else [0]) for i,row in  enumerate        (sorted_indices)]
            # next_token_ids = torch.stack([row[random.choice(selected_indices[i])] for i,row in enumerate(topk_indices)]).unsqueeze(-1)
            next_token_ids = torch.stack([row[selected_indices[i][torch.randint(0,len(selected_indices[i]),(1,)).item()]] for i,row in enumerate(topk_indices)]).unsqueeze(-1)
            
            sqs_token_ids = torch.concat([sqs_token_ids,next_token_ids],dim=-1)
            # 惩罚衰减
            penalty *= penalty_decay
            # 更新重复惩罚
            for i,p in enumerate(penalty):
                p[next_token_ids[i]] += penalty_weight
        # 去掉左padding，删除eos_token后内容
        output =list(map(lambda x:str(decoder(x[torch.nonzero(x).min():torch.where(x==eos_token_id)[0][0]] if len(torch.where(x==eos_token_id)[0]) else x[torch.nonzero(x).min():])[0]),sqs_token_ids))
            
        return output
    
    @classmethod
    def load_dataset(self,dataset_path):
        with open(dataset_path,'r') as f:
            data = json.load(f)    
            input = []
            for item in data:
                input.append(item['input'])
            self.dataset = NccDataset(input)
        f.close()
        
    @classmethod
    def run(self,output_path=None,batch_size=1,shuffle=False,*args,**kwargs):
        # save to json if output_path else return result array
        if not self.dataset:
            raise Exception('No dataset loaded')
        self.dataLoader = DataLoader(self.dataset,batch_size,shuffle)

        output = []
        for batch in tqdm(self.dataLoader):
            bacth_output = self.generate(batch['input'],*args,**kwargs)
            output += bacth_output
        
        for i,item in enumerate(output):
            output[i] = {
                "output": item
            }
        
        if output_path:
            os.makedirs(os.path.dirname(output_path),exist_ok=True)
            with open(output_path,'w+') as f:
                json.dump(output,f,indent=4)
            f.close()
        else:
            return output