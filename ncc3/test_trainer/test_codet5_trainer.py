import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
import torch
from codetf.trainer.codet5_trainer import CodeT5Seq2SeqTrainer
from codetf.data_utility.codexglue_dataset import CodeXGLUEDataset
from codetf.models import load_model_pipeline
from codetf.performance.evaluation_metric import EvaluationMetric
from codetf.data_utility.base_dataset import CustomDataset

model_class = load_model_pipeline(model_name="codet5", task="pretrained",
            model_type="plus-220M", is_eval=False, load_in_8bit=True)

dataset = CodeXGLUEDataset(tokenizer=model_class.get_tokenizer())
train, test, validation = dataset.load(subset="text-to-code")

train_dataset= CustomDataset(train[0], train[1])
test_dataset= CustomDataset(test[0], test[1])
val_dataset= CustomDataset(validation[0], validation[1])

evaluator = EvaluationMetric(metric="bleu", tokenizer=model_class.tokenizer)

# peft can be in ["lora", "prefixtuning"]
trainer = CodeT5Seq2SeqTrainer(train_dataset=train_dataset, 
                                validation_dataset=val_dataset, 
                                checkpoints_path="./checkpoints",
                                peft="lora",
                                pretrained_model_or_path=model_class.get_model(),
                                tokenizer=model_class.tokenizer)
trainer.train()
# trainer.evaluate(test_dataset=test_dataset)





