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

class CodeBertTrainer(BaseTrainer):
    def __init__(self, train_dataset, validation_dataset=None, tokenizer=None, 
                checkpoints_path="./checkpoints", pretrained_model_or_path="microsoft/codebert-base", 
                training_args=None, evaluator=None, evaluation_fn=None, peft=None):
        
        super().__init__(pretrained_model_or_path, tokenizer, train_dataset, validation_dataset,
                        checkpoints_path, pretrained_model_or_path,
                        evaluator, evaluation_fn)
        
        if training_args is None:
            self.training_args = self.get_default_codet5_hyperparameters()
        else:
            self.training_args = training_args

        self.trainer = self.init_trainer()

        if peft:
            self.peft = peft
            self.model = prepare_model_for_int8_training(self.model)
            peft_config = self.get_default_peft_config(self, peft)
            self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        rtd_labels = inputs.pop("rtd_labels")

        outputs = model(**inputs)
        logits = outputs.logits

        mlm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        hidden_states = outputs.hidden_states[-1]  # 确保输出最后的 hidden_states
        rtd_logits = hidden_states[:, :, 0]  # 这里适配 RTD 逻辑，具体实现请根据模型输出调整

        rtd_loss = F.binary_cross_entropy_with_logits(
            rtd_logits.view(-1),
            rtd_labels.float().view(-1)
        )

        total_loss = mlm_loss + rtd_loss
        return (total_loss, outputs) if return_outputs else total_loss
