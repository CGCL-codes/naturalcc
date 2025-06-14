import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset
from accelerate import Accelerator,DistributedType
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments,logging,set_seed, get_linear_schedule_with_warmup, AdamW
from omegaconf import OmegaConf
from ncc.trainer.base_trainer import BaseTrainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, AdaLoraConfig, prepare_model_for_int8_training
from ncc.utils.common.utils import get_abs_path
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sacrebleu
import os 
import inspect
class CodeBertTrainer(BaseTrainer):
    def __init__(self, train_dataset, validation_dataset=None, tokenizer=None, 
                checkpoints_path="./checkpoints", pretrained_model_or_path="microsoft/codebert-base", 
                evaluation_fn=None, training_args=None, evaluator=None, peft=None, data_collator=None):
        
        super().__init__(pretrained_model_or_path, tokenizer, train_dataset, validation_dataset,
                        checkpoints_path, pretrained_model_or_path,
                        evaluator, evaluation_fn, data_collator)
        
        self.training_args = training_args
        self.trainer = self.init_trainer()
        # print('trainer.compute_loss_func:',inspect.getsource(self.trainer.compute_loss_func))
        if peft:
            self.peft = peft
            self.model = prepare_model_for_int8_training(self.model)
            peft_config = self.get_default_peft_config(self, peft)
            self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
    
