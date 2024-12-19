export HF_ENDPOINT=https://hf-mirror.com

cd bigcode-evaluation-harness
pip install -e .
pyhton bigcode-evaluation-harness\main.py
    --model #xxx \
    --peft_model #xx \
    --tasks #humaneval \
    --allow_code_execution \
    --save_generations \
    --save_generations_path #xxx \
    --n_samples #12 \
    --metric_output_path #xxx