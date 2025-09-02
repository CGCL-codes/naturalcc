import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import os

# --- 1. 准备工作 ---

# 创建一个虚拟的 .jsonl 文件用于演示，格式与 CodeSearchNet 类似
def create_dummy_jsonl_file():
    print("创建虚拟的 'codesearchnet_sample.jsonl' 文件用于演示...")
    dummy_data = [
        {"docstring": "This function calculates factorial.", "code": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"},
        {"docstring": "Initializes a numpy array.", "code": "import numpy as np\n\narr = np.array([1, 2, 3])\nprint(arr.shape)"},
        {"docstring": "A simple calculator class.", "code": "class SimpleCalculator:\n    def add(self, a, b):\n        return a + b"},
        # 假设某一行可能缺少 docstring
        {"docstring": None, "code": "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World\");\n    }\n}"}
    ]
    with open("codesearchnet_sample.jsonl", "w", encoding="utf-8") as f:
        for entry in dummy_data:
            f.write(json.dumps(entry) + '\n')
    print("虚拟文件创建完毕。")

# --- 2. 数据加载 (已更新) ---

# 这个迭代器现在可以处理 .jsonl 文件，并从指定的多个键中提取文本
def get_text_iterator_from_jsonl(jsonl_path: str, text_keys: list):
    """
    从 .jsonl 文件中逐行读取数据，并从指定的多个键 (text_keys) 中提取文本。
    """
    print(f"从 '{jsonl_path}' 中逐行读取数据，提取键为 {text_keys} 的内容...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 解析每一行作为一个独立的 JSON 对象
                item = json.loads(line.strip())
                # 遍历需要提取的键
                for key in text_keys:
                    # 确保键存在且值不为空
                    if key in item and item[key]:
                        yield item[key]
            except json.JSONDecodeError:
                print(f"警告：跳过无法解析的行: {line.strip()}")


# --- 3. 训练分词器 (与之前相同) ---

def train_new_tokenizer(jsonl_path: str, text_keys: list, vocab_size: int, output_dir: str, special_tokens: list):
    """
    训练一个新的 BBPE 分词器并保存结果。
    """
    print("\n--- 开始训练新的分词器 ---")
    
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # 从新的 .jsonl 数据迭代器中进行训练
    text_iterator = get_text_iterator_from_jsonl(jsonl_path, text_keys)
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    
    print("--- 训练完成 ---")

    # --- 4. 保存结果 (与之前相同) ---

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    full_tokenizer_path = os.path.join(output_dir, "new_bbpe_tokenizer.json")
    tokenizer.save(full_tokenizer_path)
    print(f"完整的分词器已保存到: {full_tokenizer_path}")

    vocab = tokenizer.get_vocab()
    vocab_path = os.path.join(output_dir, "new_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"新的词汇表（用于匹配脚本）已保存到: {vocab_path}")
    print(f"\n现在你可以将 '{vocab_path}' 作为 --vocab2 参数用于你的匹配脚本了！")


if __name__ == '__main__':
    # --- 配置参数 (请根据你的情况修改) ---
    
    # 你的 CodeSearchNet .jsonl 训练数据文件路径
    # 这里我们使用脚本自动生成的虚拟文件作为示例
    TRAIN_FILE_PATH = "codesearchnet_sample.jsonl"
    
    # 【重要】指定要从每行 JSON 中提取文本的字段列表
    # 对于 CodeSearchNet，就是 ["docstring", "code"]
    TEXT_FIELD_KEYS = ["docstring", "code"]

    # 新词汇表的目标大小
    NEW_VOCAB_SIZE = 151,665
    
    OUTPUT_DIRECTORY = "new_tokenizer_output"
    # 输出目录
    QWEN_SPECIAL_TOKENS = [
         # 核心和对话
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    
    # 代码专用
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",

    # 工具调用
    "<tool_call>",
    "</tool_call>",

    # 多模态 (为了兼容性)
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>"
    ]
    
    # 运行虚拟文件创建函数（如果你有自己的 .jsonl 文件，请注释掉这一行）
    create_dummy_jsonl_file()

    # 开始训练
    train_new_tokenizer(
        jsonl_path=TRAIN_FILE_PATH,
        text_keys=TEXT_FIELD_KEYS,
        vocab_size=NEW_VOCAB_SIZE,
        output_dir=OUTPUT_DIRECTORY
    )