# CONSTANT for settings
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

MAX_HOP = None
ONLY_DEF = True
ENABLE_DOCSTRING = True
LAST_K_LINES = 1

MODEL = "codellama7b"
# MODEL = "deepseekcoder"
# MODEL = "qwencoder"
# MODEL = "starcoder2"
# MODEL = "qwen3"
MODEL = "llama3"

FILE = "call"
RAG_DIR = Path(__file__).resolve().parent
DS_BASE_DIR = RAG_DIR / "CEval"
DS_REPO_DIR = str(DS_BASE_DIR / f"{FILE}_repo")
DS_FILE = str(DS_BASE_DIR / f"{FILE}_metadata.jsonl")
DS_GRAPH_DIR = str(DS_BASE_DIR / f"{FILE}_graph")
PT_FILE = str(DS_BASE_DIR / f"{FILE}_{MODEL}_prompt.jsonl")
RESULT_DIR = RAG_DIR / "results" / MODEL / "vllm"
EVAL_FILE = str(RESULT_DIR / f"{FILE}_{MODEL}_eval.txt")
RESULT_FILE = str(RESULT_DIR / f"{FILE}_{MODEL}_result.json")
IMP_FILE = str(RESULT_DIR / f"{FILE}_{MODEL}_improved.json")
LC_PT_FILE = str(DS_BASE_DIR / f"{FILE}_langchain_prompt.jsonl")
