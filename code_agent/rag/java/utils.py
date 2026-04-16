# CONSTANT for settings
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
MAX_HOP = None
# ONLY_DEF = True
ONLY_DEF=False

ENABLE_DOCSTRING = True
LAST_K_LINES = 1

import os
# MODEL = "codellama7b"
# MODEL = "deepseekcoder"
# MODEL = "qwencoder"
# MODEL = "starcoder2"
# MODEL = "qwen3"
# MODEL = "llama3"
MODEL = "gpt51"

FILE = "javaclean"
DS_BASE_DIR = os.path.abspath("../JavaEval")
DS_REPO_DIR = os.path.join(DS_BASE_DIR, f"{FILE}_repo")
DS_FILE = os.path.join(DS_BASE_DIR, f"{FILE}_metadata.jsonl")
DS_GRAPH_DIR = os.path.join(DS_BASE_DIR, f"{FILE}_graph")
PT_FILE = os.path.join(DS_BASE_DIR, f"{FILE}_{MODEL}_prompt.jsonl")
BASE_DIR = os.path.abspath("../")
# RESULT_DIR = os.path.join(BASE_DIR, f"results/{MODEL}")
RESULT_DIR = os.path.join(BASE_DIR, f"results/{FILE}/{MODEL}/vllm/")
EVAL_FILE = os.path.join(RESULT_DIR, f"{FILE}_{MODEL}_eval.txt")
RESULT_FILE = os.path.join(RESULT_DIR, f"{FILE}_{MODEL}_result.json")
IMP_FILE = os.path.join(RESULT_DIR, f"{FILE}_{MODEL}_improved.json")

LC_PT_FILE = os.path.join(DS_BASE_DIR, f"{FILE}_langchain_prompt.jsonl")
