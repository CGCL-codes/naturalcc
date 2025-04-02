import sys, os
sys.path.append(os.path.abspath('.'))
import torch
from torch.utils.data import random_split, Subset
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from transformers import AutoProcessor,Pix2StructForConditionalGeneration,TrainingArguments,AddedToken,HfArgumentParser,Trainer
from transformers.optimization import Adafactor,get_cosine_schedule_with_warmup,get_constant_schedule_with_warmup
from my_dataset import UICoderDataset,UICoderCollater
from vars import *
from utils import smart_tokenizer_and_embedding_resize
from dataclasses import dataclass, field, asdict
import wandb
import multiprocessing
from torch.optim import SGD

torch.manual_seed(SEED)

@dataclass
class MyTrainingArguments():
    stage: int = field(default=1)
    model_name_or_path: str = field(default='/data02/models/pix2struct-large/')
    batch_size: int = field(default=1)
    output_path: str = field(default='/data02/users/lz/code/UICoder/checkpoints/')
    eval_size: int = field(default=1024)
    num_train_epochs: int = field(default=3)
    learning_rate: float = field(default=1e-4)
    save_steps:int=field(default=10000)

@dataclass
class MyDataArguments():
    name: str = field(default='ws')
    path: str = field(default='/data02/users/lz/code/UICoder/datasets/WebSight-format-parquet')
    max_length: int = field(default=2048)
    max_patches: int = field(default=2048)
    max_num: int = field(default=-1)
    preprocess: bool = field(default=True)
    make_patches_while_training: bool = field(default=True)
    
def train():    
    parser = HfArgumentParser((MyTrainingArguments,MyDataArguments))
    training_args,data_args = parser.parse_args_into_dataclasses()

    # wandb 配置
    PROJECT = 'bboxer' if training_args.stage == 1 else ('styler' if training_args.stage == 2 else 'end2end') 
    NAME = f'l{data_args.max_length}_p{data_args.max_patches}_{data_args.name}_{f"{int(data_args.max_num/1e6)}m" if data_args.max_num>=1e6 else f"{int(data_args.max_num/1e3)}k"}*{training_args.num_train_epochs}'
    wandb.init(project=PROJECT,name=NAME)

    data_args.max_num = data_args.max_num+training_args.eval_size if data_args.max_num!=-1 else -1

    training_args_2 = TrainingArguments(
        output_dir=os.path.join(training_args.output_path,f'stage{training_args.stage}/{NAME}'),
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=training_args.batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=training_args.batch_size,
        num_train_epochs=training_args.num_train_epochs,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=training_args.save_steps,
        save_total_limit=5,
        logging_strategy="steps",
        logging_steps=1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        dataloader_num_workers=max(1, int(multiprocessing.cpu_count()*0.5)),
        save_only_model=True
    )
     
    # 模型   
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = Pix2StructForConditionalGeneration.from_pretrained(training_args.model_name_or_path,is_encoder_decoder=True)
    
    smart_tokenizer_and_embedding_resize(model, processor.tokenizer, {
        'bos_token': AddedToken('<s>', rstrip=False, lstrip=False, single_word=False, normalized=True),
    })
    
    data_args = asdict(data_args)
    data_args.pop('name')

    # 数据
    dataset = UICoderDataset(**data_args,processor=processor,drop_longer=True,stage=training_args.stage)
    eval_dataset = Subset(dataset,range(0,training_args.eval_size))
    train_dataset = Subset(dataset,range(training_args.eval_size,len(dataset)))
    
    # 优化器
    #optimizer = SGD(model.parameters(), lr=0.001, nesterov=False, momentum=0.9)
    optimizer = Adafactor(model.parameters(),scale_parameter=False,relative_step=False,lr=training_args_2.learning_rate,weight_decay=training_args_2.weight_decay)
    total_steps = int(training_args_2.num_train_epochs*len(train_dataset)/training_args_2.per_device_train_batch_size/training_args_2.gradient_accumulation_steps/torch.cuda.device_count())
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=int(training_args_2.warmup_ratio*total_steps),num_training_steps=total_steps)
    
    trainer = Trainer(
        model=model,
        args=training_args_2,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        data_collator=UICoderCollater(),
        optimizers=(optimizer,scheduler)
    )

    trainer.train()

if __name__ == '__main__':
    train()
