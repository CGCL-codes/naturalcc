
import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset
from accelerate import Accelerator,DistributedType
from torch.optim.lr_scheduler import OneCycleLR
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments,logging,set_seed
from omegaconf import OmegaConf
from accelerate import Accelerator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, AdaLoraConfig, PromptEncoderConfig, PrefixTuningConfig, PromptTuningInit, PromptTuningConfig
from ncc.utils.common.utils import get_abs_path
import sacrebleu
from transformers.trainer_pt_utils import get_parameter_names
import os 

class BaseTrainer():
    
    DEFAULT_CODET5_HYPERPARAMETERS_PATH = "configs/training/codet5.yaml"
    DEFAULT_CAUSAL_LM_HYPERPARAMETERS_PATH = "configs/training/causal_lm.yaml"

    def __init__(self, model, tokenizer, train_dataset, validation_dataset=None,
                checkpoints_path="./checkpoints", pretrained_model_or_path=None,
                evaluator=None, evaluation_fn=None):
        
        self.saved_checkpoints_path = checkpoints_path
        self.create_checkpoints_path(checkpoints_path)
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        # check for evaluator and evaluation_fn, cannot co-exist
        if evaluator is not None and evaluation_fn is not None:
            raise ValueError("evaluator and evaluation_fn cannot co-exist. Please choose one.")

        if evaluator is not None:
            self.compute_metrics_fn = evaluator.compute
        elif evaluation_fn is not None:
            self.compute_metrics_fn = evaluation_fn
        else:
            self.compute_metrics_fn = None

    def init_trainer(self):
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            compute_metrics=self.compute_metrics_fn
        )

    def train(self):
        self.trainer.train()
        # self.trainer.save_model(self.saved_checkpoints_path)
    
    def evaluate(self, dataset=None):
        self.trainer.evaluate(dataset)

    def get_default_codet5_hyperparameters(self):
        codet5_hyperparameters_config = OmegaConf.load(get_abs_path(self.DEFAULT_CODET5_HYPERPARAMETERS_PATH)).hyperparameters

        training_args = TrainingArguments(
            per_device_train_batch_size=codet5_hyperparameters_config["per_device_train_batch_size"],
            gradient_accumulation_steps=codet5_hyperparameters_config["gradient_accumulation_steps"],
            num_train_epochs=codet5_hyperparameters_config["num_train_epochs"],
            warmup_steps=codet5_hyperparameters_config["warmup_steps"],
            learning_rate=codet5_hyperparameters_config["learning_rate"],
            fp16=codet5_hyperparameters_config["fp16"],
            fsdp=codet5_hyperparameters_config["fsdp"],
            sharded_ddp=codet5_hyperparameters_config["sharded_ddp"],
            logging_steps=codet5_hyperparameters_config["logging_steps"],
            evaluation_strategy=codet5_hyperparameters_config["evaluation_strategy"],
            save_strategy=codet5_hyperparameters_config["save_strategy"],
            gradient_checkpointing=codet5_hyperparameters_config["gradient_checkpointing"],
            auto_find_batch_size=codet5_hyperparameters_config["auto_find_batch_size"],
            load_best_model_at_end=codet5_hyperparameters_config["load_best_model_at_end"],
            output_dir=self.saved_checkpoints_path
        )
        # return hyperparameters_config
        return training_args
    
    def get_default_causal_lm_hyperparameters(self):
        causal_lm_hyperparameters_config = OmegaConf.load(get_abs_path(self.DEFAULT_CAUSAL_LM_HYPERPARAMETERS_PATH)).hyperparameters

        training_args = TrainingArguments(
            per_device_train_batch_size=causal_lm_hyperparameters_config["per_device_train_batch_size"],
            gradient_accumulation_steps=causal_lm_hyperparameters_config["gradient_accumulation_steps"],
            num_train_epochs=causal_lm_hyperparameters_config["num_train_epochs"],
            warmup_steps=causal_lm_hyperparameters_config["num_train_epochs"],
            learning_rate=causal_lm_hyperparameters_config["learning_rate"],
            fp16=causal_lm_hyperparameters_config["fp16"],
            fsdp=causal_lm_hyperparameters_config["fsdp"],
            sharded_ddp=causal_lm_hyperparameters_config["sharded_ddp"],
            logging_steps=causal_lm_hyperparameters_config["logging_steps"],
            evaluation_strategy=causal_lm_hyperparameters_config["evaluation_strategy"],
            save_strategy=causal_lm_hyperparameters_config["save_strategy"],
            gradient_checkpointing=causal_lm_hyperparameters_config["gradient_checkpointing"],
            auto_find_batch_size=causal_lm_hyperparameters_config["auto_find_batch_size"],
            load_best_model_at_end=causal_lm_hyperparameters_config["load_best_model_at_end"],
            output_dir=self.saved_checkpoints_path
        )
        # return hyperparameters_config
        return training_args
    
    def get_default_peft_config(self, peft_type):
        peft_config = None
        if peft_type == 'lora':
            config = OmegaConf.load(get_abs_path(self.DEFAULT_CODET5_HYPERPARAMETERS_PATH)).lora
            peft_config =  LoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            )
        elif peft_type == 'adalora':
            config = OmegaConf.load(get_abs_path(self.DEFAULT_CODET5_HYPERPARAMETERS_PATH)).adalora
            peft_config =  AdaLoraConfig(
            init_r=config.init_r,
            r=config.target_r,
            beta1=config.beta1,
            beta2=config.beta2,
            tinit=config.tinit,
            tfinal=config.tfinal,
            deltaT=config.deltaT,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type="CAUSAL_LM",
            inference_mode=config. inference_mode,
            )
        elif peft_type == 'prompt':
            config = OmegaConf.load(get_abs_path(self.DEFAULT_CODET5_HYPERPARAMETERS_PATH)).prompt
            peft_config =  PromptTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=config.num_virtual_tokens,
            )
        elif peft_type == 'p_tuning':
            config = OmegaConf.load(get_abs_path(self.DEFAULT_CODET5_HYPERPARAMETERS_PATH)).p_tuning
            peft_config =  PromptEncoderConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=config.num_virtual_tokens,
            encoder_hidden_size=config.prompt_encoder_hidden_size
            )
        elif peft_type == 'prefix':
            config = OmegaConf.load(get_abs_path(self.DEFAULT_CODET5_HYPERPARAMETERS_PATH)).prefix
            peft_config =  PrefixTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=config.num_virtual_tokens,
            encoder_hidden_size=config.prompt_encoder_hidden_size,
            prefix_projection=True,
            )
            self.model.gradient_checkpointing_disable()
        else:
            assert peft_type, "Error: Wrong type of peft."    
        return peft_config

    def create_checkpoints_path(self, checkpoints_path):
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
    
    

       
    
   