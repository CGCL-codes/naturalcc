import sys
from pathlib import Path
# sys.path.append(str(Path("").absolute().parent))
sys.path.append(str(Path(__file__).resolve().parents[2]))
import torch
from ncc.trainer.codebert_trainer import CodeBertTrainer
from ncc.utils.data_util.codesearch_dataset import CodeSearchDataset, CustomDataCollatorForMLMAndRTD, preprocess_function
from ncc.models import load_model_pipeline
from ncc.utils.data_util.base_dataset import CustomDataset
import yaml
from ncc.configs.training

def main():
    config_path = "../ncc/configs/training/codebert.yaml"
    config = load_config(config_path)
    training_args = config['hyperparameters']

    model_class = load_model_pipeline(model_name="causallm", task="pretrained",
                model_type="codegen-350M-mono", is_eval=False, load_in_8bit=False)

    data_class = CodeSearchDataset(tokenizer=model_class.get_tokenizer())
    train, validation, test = data_class.load(subset="all")

    train_dataset = data_class.get_tokenized_dataset(train)
    valid_dataset = data_class.get_tokenized_dataset(validation)
    test_dataset = data_class.get_tokenized_dataset(test)

    data_collator = CustomDataCollatorForMLMAndRTD(tokenizer=model_class.get_tokenizer(), mlm=True, mlm_probability=0.15)

    trainer = CodeBertTrainer(
        pretrained_model_or_path=model_class,
        tokenizer=model_class.get_tokenizer(),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        training_args=training_args,
        checkpoints_path="./checkpoints",
        peft=None,
    )

    trainer.train()


if __name__ == "__main__":
    main()