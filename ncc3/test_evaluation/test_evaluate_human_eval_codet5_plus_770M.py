import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from ncc3.models import load_model_pipeline
from ncc3.utils.data_util.human_eval_dataset import HumanEvalDataset
from codetf.evaluate.model_evaluator import ModelEvaluator
from torch.utils.data import TensorDataset
import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_class = load_model_pipeline(model_name="codet5", task="pretrained",
            model_type="plus-770M-python", is_eval=True)

dataset = HumanEvalDataset(tokenizer=model_class.get_tokenizer())
prompt_token_ids, prompt_attention_masks, references= dataset.load()

problems = TensorDataset(prompt_token_ids, prompt_attention_masks)

evaluator = ModelEvaluator(model_class)
avg_pass_at_k = evaluator.evaluate_pass_k(problems=problems, unit_tests=references, sequences_per_chunk=20, num_workers=10)
print("Pass@k: ", avg_pass_at_k)