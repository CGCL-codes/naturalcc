from ncc2.tasks import NccTask
from ncc2.models.hf import HFTokenizer,HFBuilder,hf_archs
from ncc2.tasks import register_task
from ncc2.tasks.utils.task_registry import TaskRegistry
from dataclasses import dataclass
from ncc2.models.utils.arch_registry import ArchitectureRegistry
from ncc2.data.text import TextTokenizer
from torch.utils.data import DataLoader
from ncc2.models.sequence import SequenceBatch
from ncc2.nn.padding import PaddingMask
from tqdm import tqdm
from ncc2.data.dataset import NccDataset
import torch.nn.functional as F
import torch
import json
import os

@dataclass
class SummarizationTaskConfig:
    archs: ArchitectureRegistry
    model_name: str
    builder: object
    tokenizer_cls: TextTokenizer     
    
summarization_tasks = TaskRegistry[SummarizationTaskConfig]('summarization')
summarization_task = summarization_tasks.marker
    
@summarization_task('auto')
def auto() -> SummarizationTaskConfig:
    return SummarizationTaskConfig(
        archs=hf_archs,
        model_name='auto',
        builder=HFBuilder,
        tokenizer_cls=HFTokenizer
    )

@register_task('summarization')
class SummarizationTask(NccTask):
    @classmethod
    def __init__(self,config: SummarizationTaskConfig=None,task_name: str=None,*args,**kwargs):
        if not config:
            if task_name:
                config = summarization_tasks.get_config(task_name)
            else:
                raise ValueError('Config or task_name are needed')
        super().__init__(*args,**vars(config),**kwargs)
        
    @classmethod
    def generate(self,input,*args,tokenizer=None,**kwargs):
        if not tokenizer:
            if self.tokenizer:
                tokenizer = self.tokenizer
            else:
                raise Exception('No tokenizer loaded')
        encoder = tokenizer.create_encoder()
        decoder = tokenizer.create_decoder()
        
        if isinstance(input,str):
            sqs = [input]
        elif isinstance(input,list):
            sqs = input
        else:
            raise ValueError('Input should be str|list[str]')
        
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
        
    @classmethod
    def from_pretrained(self, ckpt_folder):
        self.model = self.model.from_pretrained(ckpt_folder,device_map={"":self.device},trust_remote_code=True)
        self.tokenizer = self.tokenizer_cls(ckpt_folder)