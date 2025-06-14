import sys
from pathlib import Path
# sys.path.append(str(Path("").absolute().parent))
sys.path.append(str(Path(__file__).resolve().parents[2]))
import torch
from ncc.trainer.codebert_trainer import CodeBertTrainer
from ncc.utils.data_util.codesearch_dataset import CodeSearchDataset, load_tokenized_dataset
from ncc.models import load_model_pipeline
from ncc.utils.data_util.base_dataset import CustomDataset
import yaml
from pathlib import Path
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
import inspect
from torch.utils.data import DataLoader
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
def main():
    config_path = Path(__file__).resolve().parents[2] / "ncc/configs/training/codebert.yaml"
    config = load_config(config_path)
    hyperparameters = config['hyperparameters']

    # 创建 TrainingArguments 实例
    training_args = TrainingArguments(
        output_dir=hyperparameters['output_dir'],
        evaluation_strategy=hyperparameters['eval_strategy'],
        eval_steps=hyperparameters['eval_steps'],
        save_steps=hyperparameters['save_steps'],
        save_total_limit=hyperparameters['save_total_limit'],
        logging_steps=hyperparameters['logging_steps'],
        learning_rate=float(hyperparameters['learning_rate']),
        per_device_train_batch_size=hyperparameters['per_device_train_batch_size'],
        per_device_eval_batch_size=hyperparameters['per_device_eval_batch_size'],
        num_train_epochs=hyperparameters['num_train_epochs'],
        weight_decay=hyperparameters['weight_decay'],
        warmup_steps=hyperparameters['warmup_steps'],
        logging_dir=hyperparameters['logging_dir'],
        save_strategy=hyperparameters['save_strategy'],
        bf16=hyperparameters['bf16'],
        report_to=hyperparameters['report_to'],
        seed=hyperparameters['seed']
    )

    model = load_model_pipeline(
        model_name="bert-pretrain",
        task="pretrained",
        model_type="codebert-base",
        is_eval=False,
        load_in_8bit=False
    )

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # 从本地加载预处理后的数据集
    save_path = Path("/mnt/silver/tanlei/datasets/codebert")
    train_dataset = load_tokenized_dataset(save_path / "train")
    valid_dataset = load_tokenized_dataset(save_path / "validation")

    trainer = CodeBertTrainer(
        pretrained_model_or_path=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=valid_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15),
        training_args=training_args,
        checkpoints_path="./checkpoints",
        peft=None,
    )

    trainer.train()

if __name__ == "__main__":
    main()