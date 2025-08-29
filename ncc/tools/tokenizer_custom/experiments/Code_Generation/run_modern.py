import logging
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys
from datasets import load_dataset
# from model_modern import ModernSeq2Seq # 从您更新的文件中导入
import bleu # 假设 bleu.py 在同一目录下
import json
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    set_seed,RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)

from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)

### modified part
from utils import modify_mapping
### end of modified part

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
                 'decoder': (GPT2Config, AutoModelForCausalLM, AutoTokenizer),
                 }

# --- 设置日志 ---
# (现代化改进: 使用更标准的日志设置方式)
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     # stream=sys.stdout,
#     level=logging.INFO,
# )

# 参数管理 (HfArgumentParser & dataclasses):
# 我们用 Python 的 dataclass 定义了三组清晰的参数：ModelArguments, DataTrainingArguments, 和 Seq2SeqTrainingArguments (Hugging Face 自带)。
# HfArgumentParser 可以直接将命令行参数解析到这些 dataclass 对象中，代码更整洁，且自带类型检查。

# --- 定义脚本参数 ---
# (现代化改进: 使用 dataclass 定义参数，更清晰、类型安全)
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    # use_fast=True (默认)：这行代码告诉程序，如果存在快速版本，请优先使用它。快速分词器在处理大量文本时速度要快得多，并且支持一些高级功能，比如并行处理。

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: str = field(metadata={"help": "The input training data file (a jsonl file)."})
    validation_file: str = field(metadata={"help": "An optional input evaluation data file (a jsonl file)."})
    test_file: str = field(metadata={"help": "An optional input test data file (a jsonl file)."})
    max_source_length: int = field(
        default=256,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    max_target_length: int = field(
        default=256,
        metadata={"help": "The maximum total target sequence length after tokenization."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

# ==================== 在这里添加新代码 ====================
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import os

class CustomSaveCallback(TrainerCallback):
    """
    一个自定义的回调函数，用于复刻旧脚本的保存逻辑：
    1. checkpoint-last: 每个评估周期都保存。
    2. checkpoint-best-ppl: 当 eval_loss 创下新低时保存。
    3. checkpoint-best-bleu: 当 eval_bleu 创下新高时保存。
    """
    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')
        self.best_bleu = float('-inf')

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        # 从评估结果中获取最新的 metrics
        # state.log_history 中最后一个元素就是本次的评估结果
        metrics = state.log_history[-1]

        # 确保 metrics 字典里有我们需要的值
        current_loss = metrics.get("eval_loss")
        current_bleu = metrics.get("eval_bleu")

        # --- 1. 保存最新的模型 (checkpoint-last) ---
        # 这个逻辑无条件执行
        last_checkpoint_dir = os.path.join(args.output_dir, "checkpoint-last")
        model.save_pretrained(last_checkpoint_dir)
        # 您也可以使用 trainer.save_model(last_checkpoint_dir)，效果类似
        print(f"Saved last model checkpoint to {last_checkpoint_dir}")

        # --- 2. 根据 eval_loss 保存最佳模型 (checkpoint-best-ppl) ---
        if current_loss is not None and current_loss < self.best_loss:
            self.best_loss = current_loss
            best_ppl_checkpoint_dir = os.path.join(args.output_dir, "checkpoint-best-ppl")
            model.save_pretrained(best_ppl_checkpoint_dir)
            print(f"New best PPL model found! Saved to {best_ppl_checkpoint_dir} (Loss: {current_loss:.4f})")

        # --- 3. 根据 eval_bleu 保存最佳模型 (checkpoint-best-bleu) ---
        if current_bleu is not None and current_bleu > self.best_bleu:
            self.best_bleu = current_bleu
            best_bleu_checkpoint_dir = os.path.join(args.output_dir, "checkpoint-best-bleu")
            model.save_pretrained(best_bleu_checkpoint_dir)
            print(f"New best BLEU model found! Saved to {best_bleu_checkpoint_dir} (BLEU: {current_bleu})")

# ==================== 新代码添加结束 ====================

def main():
    
    # --- 解析参数 ---
    #旧方法的  parser = argparse.ArgumentParser  现代化改进: 使用 HfArgumentParser 解析 dataclass 参数)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    # 旧方法: parse_args() 新方法arse_args_into_dataclasses() 就像一个智能分拣系统。它自动读取信件，并把它们分门别类地放进三个贴好标签的文件夹里：“模型相关”（model_args）、“数据相关”（data_args）和“训练流程”（training_args）
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # ==================== 代码修改开始 ====================
    # --- 设置日志 (同时输出到文件和控制台) ---
    # 确保输出目录存在
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    log_file_path = os.path.join(training_args.output_dir, "training_run.log")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        # handlers 参数让日志同时输出到两个地方
        handlers=[
            # 1. 输出到文件，并强制使用 utf-8 编码
            logging.FileHandler(log_file_path, encoding='utf-8'),
            # 2. 输出到标准输出（即控制台）
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    # ==================== 代码修改结束 ====================

    # --- 在这里添加打印所有参数的代码 ---
    # 只有主进程（rank 0）或非分布式训练时（rank -1）才打印
    if training_args.local_rank in [-1, 0]:
        print("\n" + "="*50)
        print(" " * 15, "PARSED ARGUMENTS")
        print("="*50)
        
        print("\n--- Model Arguments ---")
        print(model_args)
        
        print("\n--- Data Arguments ---")
        print(data_args)
        
        print("\n--- Training Arguments ---")
        print(training_args)
        
        print("\n" + "="*50 + "\n")
    # --- 添加结束 ---

    # --- 设置日志和随机种子 ---
    log_level = training_args.get_process_log_level()
    # 主进程 (GPU 0) 的 log_level 是 INFO，所以它会打印所有重要的训练信息（进度、损失、评估结果等）。
    # 其他进程 (GPU 1, 2, 3) 的 log_level 是 WARNING，所以它们只会打印警告或错误信息，而不会打印常规的 INFO 级日志。
    logger.setLevel(log_level)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    # --- 加载模型和分词器 ---
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )


    # 新方法将这个过程拆分成了清晰的两步。第一步使用 AutoConfig，它的唯一目的就是去读取 config.json 文件，并把它加载成一个 config 对象。这一步完全不涉及模型的权重。
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        padding_side='left'
    )
    # tokenizer.padding_side = "left"  # ✅ 左对齐 padding
    # 默认 tokenizer padding 可能是 right，而 decoder-only 模型在生成时会从左往右预测，如果右 padding，attention mask 可能会错误，导致生成结果异常或不一致。

    # (现代化改进: 自动处理特殊 token，更健壮)
    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.pad_token 操作的是字符串 (string)。
    # tokenizer.pad_token_id 操作的是整数ID (integer ID)。
    # 在 Hugging Face 的 tokenizer 内部，这两个属性是双向绑定的。修改其中一个，另一个通常会自动更新
        # config.pad_token_id = config.eos_token_id

    # 2. 解决 BOS 和 EOS token 的同步问题（为了代码的健壮性）
    # 确保 config 和 tokenizer 对起始和结束符的认知也完全一致
    config.pad_token_id = tokenizer.pad_token_id
    tokenizer.bos_token_id = config.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    # tokenizer_vocab_size = len(tokenizer)
    # print(f"Tokenizer Vocabulary Size: {tokenizer_vocab_size}")
    # model_embedding_size = hf_model.get_input_embeddings().weight.shape[0]
    # print(f"Model Embedding Layer Size1: {model_embedding_size}")    

    # 调整 token embedding 大小以匹配 tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # model_embedding_size = hf_model.get_input_embeddings().weight.shape[0]
    # print(f"Model Embedding Layer Size2: {model_embedding_size}")   
    

    # ✅ 在这里检查 embedding 大小

    # ==================== 在这里插入“配置健全性检查”代码 ====================
    if training_args.local_rank in [-1, 0]:
        embedding_size = model.get_input_embeddings().weight.shape[0]
        print(f"1111Embedding size: {embedding_size}")
        
        print("--- Running Model Config Sanity Check ---")
        vocab_size = len(tokenizer)
        print(f"Vocabulary Size: {vocab_size}")
        
        # 检查所有关键的特殊 token ID
        # 我们同时检查 config 和 tokenizer 以确保它们同步
        special_ids = {
            "config.pad_token_id": config.pad_token_id,
            "config.bos_token_id": config.bos_token_id,
            "config.eos_token_id": config.eos_token_id,
            "tokenizer.pad_token_id": tokenizer.pad_token_id,
            "tokenizer.bos_token_id": tokenizer.bos_token_id,
            "tokenizer.eos_token_id": tokenizer.eos_token_id,
        }
        
        ok = True
        for name, token_id in special_ids.items():
            print(f"Checking {name}: {token_id}")
            if token_id is not None and token_id >= vocab_size:
                print(f"!!!!!!!! FATAL CONFIG ERROR: {name} ({token_id}) is out of bounds for vocab size ({vocab_size})! !!!!!!!!")
                ok = False
                
        if ok:
            print(">>> Config Sanity Check PASSED. All special token IDs are valid.")
        else:
            print(">>> Config Sanity Check FAILED. Found invalid special token IDs.")

        print("--- Config Sanity Check Finished ---")
    # ========================== 检查代码结束 ==========================


    # 数据处理 (datasets 库):
    # 放弃了手动读写文件和创建 Example 对象的旧方法。
    # 使用 load_dataset 直接从 jsonl 文件加载数据，这个过程是可缓存和多进程的，处理大数据集时速度极快。
    # preprocess_function 集中处理了所有的分词和标签创建逻辑。注意：我们采用了为 Causal LM 设计的特殊标签 mask 方式，只计算 target 部分的 loss，这是微调这类模型的关键。

    # --- 加载和预处理数据 ---
    # (现代化改进: 使用 Hugging Face `datasets` 库，高效、可缓存)
    raw_datasets = load_dataset('json', data_files={'train': data_args.train_file, 'validation': data_args.validation_file, 'test': data_args.test_file})

    # # 文本转换为模型可用的数字特征 (Tokenization)
    # # 旧方法：手动循环处理
    # def preprocess_function(examples):
    #     # 原始数据格式是 'nl' (source) 和 'code' (target)
    #     # inputs = [ex for ex in examples['nl']]
    #     # targets = [ex for ex in examples['code']]
    #     inputs = examples['nl']
    #     targets = examples['code']

    #     # 格式化输入: "source_text <bos> target_text <eos>"
    #     # 这是为 Causal LM 微调成 Seq2Seq 任务的常见做法
    #     full_inputs = [f"{inp} {tokenizer.bos_token} {tgt}{tokenizer.eos_token}" for inp, tgt in zip(inputs, targets)]
        
    #     # 对完整输入进行分词
    #     model_inputs = tokenizer(
    #         full_inputs, 
    #         max_length=data_args.max_source_length + data_args.max_target_length, 
    #         # padding="max_length", # Trainer 会处理好 padding
    #         truncation=True
    #         # truncation=True + max_length：负责处理过长的序列（把长的砍短）。
    #     )
    #     # model_inputs['input_ids']和 model_inputs['attention_mask'] 是 tokenizer 返回的两个关键字段。

    #     # 创建 labels，就是 input_ids 的一个副本
    #     model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]
        
    #     # (重要) 为了计算损失，我们需要 mask 掉输入部分(source)的 labels
    #     # 这样模型就不会被训练去预测它已经看到的部分
    #     source_tokenized = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True)
    #     for i in range(len(model_inputs['labels'])):
    #         source_len = len(source_tokenized['input_ids'][i]) + 1 # +1 for bos_token
    #         # 将 source 和 bos_token 部分的 label 设为 -100，使其在损失计算中被忽略
    #         model_inputs['labels'][i][:source_len] = [-100] * source_len

    #         # # ==================== 新增修正逻辑 ====================
    #         # # 注释掉了，因为data_collator会自动动态、补全长度和padding
    #         # # 遍历 attention_mask，如果 mask 值为 0 (代表是 padding)，
    #         # # 就将对应位置的 label 设为 -100
    #         # current_labels = model_inputs['labels'][i]
    #         # current_mask = model_inputs['attention_mask'][i]
    #         # model_inputs['labels'][i] = [
    #         #     -100 if mask == 0 else label
    #         #     for mask, label in zip(current_mask, current_labels)
    #         # ]
    #         # # =======================================================

    #     return model_inputs
    

    # ==================== 修改后的预处理函数 ====================
    def preprocess_function(examples, mapping=None):
        # inputs = [ex for ex in examples['nl']]
        # targets = [ex for ex in examples['code']]
        inputs = examples['nl']
        targets = examples['code']

        # 1. 分别对 source 和 target 进行分词
        source_tokenized = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True)
        target_tokenized = tokenizer(targets, max_length=data_args.max_target_length, truncation=True)

        model_inputs = {'input_ids': [], 'attention_mask': [], 'labels': []}
        
        for i in range(len(inputs)):
            # 这个循环是**“重体力劳动”**。在每一次迭代中，它都在执行多个创建新列表、拼接列表、计算等相对耗时的 Python 操作。几乎所有的核心数据处理逻辑都发生在这个循环内部。当处理上百万样本时，这些重复的“重体力”操作累加起来，就会变得非常非常慢。
            source_ids = source_tokenized['input_ids'][i]
            target_ids = target_tokenized['input_ids'][i]

            ### modified part
            # 2. 应用 modify_mapping 对 ID 进行校正
            if mapping:
                source_ids = modify_mapping(source_ids, mapping)
                target_ids = modify_mapping(target_ids, mapping)
            ### end of modified part

            # 3. 拼接 source, bos, target, eos
            input_ids = source_ids + [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]
            
            # 4. 创建 labels，并将 source 和 bos 部分用 -100 掩码
            labels = [-100] * (len(source_ids) + 1) + target_ids + [tokenizer.eos_token_id]

            # 5. 填充和截断到总的最大长度
            total_max_length = data_args.max_source_length + data_args.max_target_length
            
            # 填充
            # pad_len = total_max_length - len(input_ids)
            # if pad_len > 0:
            #     input_ids = [tokenizer.pad_token_id] * pad_len + input_ids
            #     labels = [-100] * pad_len + labels
            
            # 截断
            if len(input_ids) > total_max_length:
                input_ids = input_ids[-total_max_length:]
                labels = labels[-total_max_length:]

            attention_mask = [1] * len(input_ids)

            model_inputs['input_ids'].append(input_ids)
            model_inputs['attention_mask'].append(attention_mask)
            # 旧代码中 attention_mask = [1] * len(input_ids) 被删除了。
            # 效果：这个任务被完全交给了 DataCollator。DataCollator 在进行动态填充时，会自动生成正确、对应的 attention_mask，代码更简洁且不易出错。
            model_inputs['labels'].append(labels)
            
        return model_inputs
    # ==========================================================

    # 应用预处理 将文本转换为模型可用的数字特征 (Tokenization)
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,# 开启批处理
        num_proc=data_args.preprocessing_num_workers,# 开启多进程
        remove_columns=raw_datasets["train"].column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

        # ==================== 在这里添加新代码 ====================
    # 只有主进程才打印，避免在多GPU训练时重复输出
    if training_args.local_rank in [-1, 0]:
        print("\n" + "="*50)
        print(" " * 10, "TOKENIZED DATASET SIZES")
        print("="*50)
        for split, dataset in tokenized_datasets.items():
            print(f" -> Number of samples in '{split}' split: {len(dataset)}")
        print("="*50 + "\n")
    # ==================== 新代码添加结束 ====================


    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]


    # 数据整理 (DataCollatorForSeq2Seq):
    # 不再需要手动在 Dataset 中对每个样本进行 padding。
    # 旧方法：静态 Padding  在 convert_examples_to_features 函数中，所有样本都被预先填充到了一个固定的、全局的最大长度 (block_size)。
    # DataCollator 会在每个批次形成时，动态地将该批次内的样本 padding 到最长序列的长度，而不是全局最大长度。这能有效节省内存和计算资源。
    
    # --- 定义数据整理器和评估指标 ---
    # (现代化改进: 使用 DataCollatorForSeq2Seq 自动处理 padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100, # 确保 padding 的 label 被忽略
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # 评估 (compute_metrics):
    # 将评估逻辑封装在一个 compute_metrics 函数里。
    # Trainer 在评估或预测时，会自动调用这个函数，传入模型的生成结果 (preds) 和真实标签 (labels)。
    # 这使得评估逻辑与训练循环解耦，更加清晰。
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # HuggingFace Trainer 在生成时会自动处理 padding，所以 pred 可能比 label 长
        # 我们需要用 -100 替换 padding token ID，以便正确解码
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # 对于 labels，同样需要处理 -100
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 计算 BLEU
        # 假设原始的 bleu.py 脚本可用
        # 我们需要将解码后的文本写入临时文件以供 bleu 脚本使用
        output_prediction_file = os.path.join(training_args.output_dir, "tmp_predictions.txt")
        gold_file = os.path.join(training_args.output_dir, "tmp_gold.txt")
        
        predictions_for_bleu = []
        accs = []
        with open(output_prediction_file, "w") as writer, open(gold_file, "w") as gold_writer:
            for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
                predictions_for_bleu.append(f"{idx}\t{pred}")
                writer.write(f"{idx}\t{pred}\n")
                gold_writer.write(f"{idx}\t{label}\n")
                # XMatch
                accs.append(pred.strip() == label.strip())


        (goldMap, predictionMap) = bleu.computeMaps(predictions_for_bleu, gold_file)
        bleu4  = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)

        # XMatch
        xmatch = round(np.mean(accs)*100, 4)

        result = {"eval_bleu": bleu4, "eval_xmatch": xmatch}
        return result

    # 训练核心 (Seq2SeqTrainer):
    # 完全替代了手动训练循环。所有关于 epoch, batch, optimizer, scheduler, scaler (for fp16), DDP 的设置和调用，现在都由 Trainer 自动处理。
    # 你只需要在 Seq2SeqTrainingArguments (或通过命令行) 中定义超参数，如学习率、批大小、梯度累积步数等。
    # trainer.train(), trainer.evaluate(), trainer.predict() 三个命令就完成了所有核心工作。

    # --- 初始化 Trainer ---
    # (现代化改进: 核心步骤，用 Trainer 替代所有手动训练循环)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[CustomSaveCallback()] # 将自定义回调函数实例添加到列表中
    )
    # 如果 training_args.predict_with_generate 是 True（开启）：
    # 那么 Trainer 的 compute_metrics 参数就会被设置为你定义好的 compute_metrics 函数。
    # 这等于告诉 Trainer：“在评估阶段，请使用模型的 .generate() 方法来生成实际的、人类可读的文本（比如一个完整的句子或代码片段）。”
    # 结果：因为模型生成了真实的文本，所以我们理应使用 compute_metrics 函数来比较“模型生成的文本”和“标准答案文本”，从而计算出 BLEU 分数这类指标。
    
    # 如果 training_args.predict_with_generate 是 False（关闭）：
    # 那么 Trainer 的 compute_metrics 参数就会被设置为 None，意味着不进行自定义指标的计算。

    # --- 开始训练、评估、测试 ---
    if training_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        trainer.save_model()  # 保存最终模型
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=data_args.max_target_length, num_beams=training_args.generation_num_beams)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset, 
            metric_key_prefix="predict", 
            max_length=data_args.max_target_length, 
            num_beams=training_args.generation_num_beams
        )
        metrics = predict_results.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            # 这是一个非常重要的检查，专门用于分布式训练。在多 GPU 训练中，每个进程都会在自己的数据子集上进行预测。为了避免每个 GPU 进程都去写同一个文件导致冲突和内容重复，这个条件确保了**只有主进程（rank 0）**负责最终的解码和文件写入工作。
            if training_args.predict_with_generate:

                predictions_ids = predict_results.predictions
                
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))


if __name__ == "__main__":
    main()




