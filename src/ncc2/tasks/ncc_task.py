from ncc2.models.utils.arch_registry import ArchitectureRegistry
from ncc2.models.utils.checkpoint_loader import load_checkpoint,upgrade_fairseq_checkpoint,convert_model_state_dict
from ncc2.nn.position_encoder import RotaryEncoder
import torch
import os
from tqdm import tqdm
from ncc2.data.text import TextTokenizer
from ncc2.models.sequence import SequenceBatch
from ncc2.nn.padding import PaddingMask
from ncc2.data.dataset import NccDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json

class NccTask():
    @classmethod
    def __init__(self,archs:ArchitectureRegistry,model_name:str,tokenizer_cls:TextTokenizer,builder,convertor:dict={},device=None):
        if not model_name in archs.names():
            raise ValueError("Arch {} don't includes model {}".format(archs.model_type,model_name))
        if not device:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("CUDA is available. Using GPU.")
            else:
                device = torch.device("cpu")
                print("CUDA is not available. Using CPU.")
        self.device = device
        self.convertor = convertor
        self.tokenizer_cls = tokenizer_cls
        self.load_model(archs,model_name,builder,device)
        
    @classmethod  
    def load_model(self,archs:ArchitectureRegistry,model_name:str,builder,device):
        self.model_name = model_name
        self.config = archs.get_config(model_name)
        self.builder = builder(self.config,device=device)
        self.model = self.builder.build_model()

    @classmethod    
    def load_state(self,ckpt_folder):
        ckpt_files = [f for f in os.listdir(ckpt_folder) if f.endswith('.pth')]
        ckpt = {}
        if not ckpt_files:
            raise FileExistsError('No *.pth found in {}'.format(ckpt_folder))
        for ckpt_file in tqdm(ckpt_files):
            ckpt_path = os.path.join(ckpt_folder, ckpt_file)
            checkpoint = torch.load(ckpt_path)
            for key in checkpoint:
                ckpt[key] = checkpoint[key]
        ckpt2 = {}
        for key in self.convertor:
            ckpt2[self.convertor[key]] = ckpt[key]
        del ckpt
        self.model.load_state_dict(ckpt2)

    @classmethod
    def load_tokenizer(self,ckpt_folder):
        self.tokenizer = self.tokenizer_cls('{}/tokenizer.model'.format(ckpt_folder))
          
    @classmethod
    def from_pretrained(self, ckpt_folder):
        if '_hf' in self.task_name:
            self.model = self.model.from_pretrained(ckpt_folder,device_map={"":self.device},trust_remote_code=True)
            self.tokenizer = self.tokenizer_cls(ckpt_folder)
        else:
            self.load_state(ckpt_folder)
        self.load_tokenizer(ckpt_folder)        
     
    @classmethod
    def generate(self,input,max_length=100,top_k=10,top_p=0.95,temperature=0.2,penalty_weight=0.5,penalty_decay=0.95,seed=None,bos_token_id=1,eos_token_id=2,unkown_token_id=0,tokenizer=None):
        if not tokenizer:
            if self.tokenizer:
                tokenizer = self.tokenizer
            else:
                raise Exception('No tokenizer loaded')
        if seed:
            torch.manual_seed(seed)
        encoder = tokenizer.create_encoder()
        decoder = tokenizer.create_decoder()
        
        sqs = self.preprocess(input)
        
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
            for i,st in enumerate(sqs_token_ids):
                if len(st) < sqs_max_length:
                    sqs_token_ids[i] = [unkown_token_id]*(sqs_max_length-len(st)) + st
        if max_length == 0:
            max_length = sqs_max_length+1
        sqs_token_ids = torch.tensor(sqs_token_ids,dtype=torch.int64).to(self.device)                    
        while True:
            # 限制生成长度   
            sqs_max_length = max([len(x) for x in sqs_token_ids])  
            if sqs_max_length >= max_length:
                break
            # 长度对齐，左填充padding          
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
    def generate_hf(self,input,*args,tokenizer=None,**kwargs):
        if not tokenizer:
            if self.tokenizer:
                tokenizer = self.tokenizer
            else:
                raise Exception('No tokenizer loaded')
        encoder = tokenizer.create_encoder()
        decoder = tokenizer.create_decoder()
        
        sqs = self.preprocess(input)
        
        x = encoder(sqs,return_tensors='pt').to(self.device)
        output = self.model.generate(x['input_ids'],attention_mask=x['attention_mask'],*args,**kwargs)
        
        return decoder(output)
    
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
    def preprocess(self,input):
        if isinstance(input,str):
            sqs = [input]
        elif isinstance(input,list):
            sqs = input
        else:
            raise ValueError('Input should be str|list[str]')
        return sqs
    
    @classmethod
    def run(self,output_path=None,batch_size=1,shuffle=False,*args,**kwargs):
        # save to json if output_path else return result array
        if not self.dataset:
            raise Exception('No dataset loaded')
        self.dataLoader = DataLoader(self.dataset,batch_size,shuffle)

        output = []
        for batch in tqdm(self.dataLoader):
            if '_hf' in self.task_name:
                batch_output = self.generate_hf(batch['input'],*args,**kwargs)
            else:
                batch_output = self.generate(batch['input'],*args,**kwargs)
            output += batch_output
        
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
    
