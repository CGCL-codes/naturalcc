import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM
from ncc.models.base_model import BaseModel
from ncc.utils.common.registry import registry
from accelerate import Accelerator
from collections import defaultdict
from tqdm import tqdm
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download

@registry.register_model("bert-pretrain")
class BertForPretrain(BaseModel):
    MODEL_DICT = "configs/training/codebert.yaml"
    def __init__(self, model, model_config, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_prediction_length = model_config["max_prediction_length"]
        self.rtd_head = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
    @classmethod
    def init_tokenizer(cls, model):
        tokenizer = AutoTokenizer.from_pretrained(model)
        return tokenizer
    
    @classmethod
    def load_huggingface_model_from_config(model_class, model_config, load_in_8bit=False, load_in_4bit=False, weight_sharding=False):
        checkpoint = model_config["huggingface_url"]

        if load_in_8bit and load_in_4bit:
            raise ValueError("Only one of load_in_8bit or load_in_4bit can be True. Please choose one.")
        
        if weight_sharding:
            try:
                # Try to download and load the json index file
                weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")
            except Exception:
                try:
                    # If that fails, try to download and load the bin file
                    weights_location = hf_hub_download(checkpoint, "pytorch_model.bin.index.json")
                except Exception as e:
                    # If both fail, raise an error
                    raise Exception(f"Failed to download weights: {str(e)}")
                    
            config = AutoConfig.from_pretrained(checkpoint)
            with init_empty_weights():
                model = AutoModelForMaskedLM.from_config(config)

            model.tie_weights()            
            model = load_checkpoint_and_dispatch(
                model, weights_location, device_map="auto", no_split_module_classes=["GPTJBlock"]
            )
        else:
            if load_in_8bit:
                model = AutoModelForMaskedLM.from_pretrained(checkpoint, 
                                            load_in_8bit=load_in_8bit, 
                                            low_cpu_mem_usage=True,
                                            device_map="auto")
            elif load_in_4bit:
                model = AutoModelForMaskedLM.from_pretrained(checkpoint, 
                                            load_in_4bit=load_in_4bit, 
                                            low_cpu_mem_usage=True,
                                            device_map="auto")
            else:
                model = AutoModelForMaskedLM.from_pretrained(checkpoint, 
                                            low_cpu_mem_usage=True,
                                            device_map="auto")


        tokenizer = model_class.init_tokenizer(model_config["tokenizer_url"])
        
        return model_class(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer
        )
   
    def forward(self, 
               input_ids: torch.Tensor, 
               attention_mask: torch.Tensor,
               labels: torch.Tensor
               ):

        # MLM任务计算
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True, 
            return_dict=True
        )
        return {
            "loss": outputs.loss,
            "hidden_states": outputs.hidden_states[-1],
            "logits": outputs.logits
        }

    def predict(self, sources):
        input_for_net = [' '.join(source.strip().split()).replace('\n', ' ') for source in sources]
        output = self.forward(input_for_net)
        return output
    
@registry.register_model("bert")
class BertModel(BaseModel):

    MODEL_DICT = "configs/inference/bert.yaml"
    def __init__(self, model, model_config, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_prediction_length = model_config["max_prediction_length"]

    @classmethod
    def init_tokenizer(cls, model):
        tokenizer = AutoTokenizer.from_pretrained(model)
        return tokenizer
    
    @classmethod
    def load_model_from_config(model_class, model_config, load_in_8bit=False, load_in_4bit=False, weight_sharding=False):
        checkpoint = model_config["huggingface_url"]

        if load_in_8bit and load_in_4bit:
            raise ValueError("Only one of load_in_8bit or load_in_4bit can be True. Please choose one.")
        
        if weight_sharding:
            try:
                # Try to download and load the json index file
                weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")
            except Exception:
                try:
                    # If that fails, try to download and load the bin file
                    weights_location = hf_hub_download(checkpoint, "pytorch_model.bin.index.json")
                except Exception as e:
                    # If both fail, raise an error
                    raise Exception(f"Failed to download weights: {str(e)}")
                    
            config = AutoConfig.from_pretrained(checkpoint)
            with init_empty_weights():
                model = AutoModel.from_config(config)

            model.tie_weights()            
            model = load_checkpoint_and_dispatch(
                model, weights_location, device_map="auto", no_split_module_classes=["GPTJBlock"]
            )
        else:
            if load_in_8bit:
                model = AutoModel.from_pretrained(checkpoint, 
                                            load_in_8bit=load_in_8bit, 
                                            device_map="auto")
            elif load_in_4bit:
                model = AutoModel.from_pretrained(checkpoint, 
                                            load_in_4bit=load_in_4bit, 
                                            device_map="auto")
            else:
                model = AutoModel.from_pretrained(checkpoint, 
                                            device_map="auto")


        tokenizer = model_class.init_tokenizer(model_config["tokenizer_url"])
        
        return model_class(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer
        )
   
    def forward(self, sources):
        encoding = self.tokenizer(sources, return_tensors='pt')
        input_ids = encoding.input_ids.to(self.device)
        generated_ids = self.model(input_ids)

        # predictions = self.tokenizer.batch_decode(generated_ids, truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
        return generated_ids

    def predict(self, sources):
        input_for_net = [' '.join(source.strip().split()).replace('\n', ' ') for source in sources]
        output = self.forward(input_for_net)
        return output