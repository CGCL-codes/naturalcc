#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml

from transformers import AutoTokenizer
import torch
import tiktoken
import attridict


class CModelTokenizer:
    # [maint-note:40598c7a3c3e]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 model：类型由调用方与函数体约束。
    # 写入目标：self.model。
    # 写入目标：self.config。
    # 直接调用：attridict。
    # 直接调用：yaml.load。
    # 直接调用：open。
    # 直接调用：self._set_tokenizer。
    # [/maint-note:40598c7a3c3e]
    def __init__(self, model):
        self.model = model
        self.config = attridict(
            yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
        )
        self._set_tokenizer()

    # [maint-note:3910a2ca9a16]
    # 维护导航：_set_tokenizer；下列为本函数源码中的接口与路径索引。
    # 分支条件：self.model == 'codegen'。
    # 分支条件：self.model == 'codegen25'。
    # 分支条件：self.model == 'santacoder'。
    # 分支条件：self.model == 'starcoder'。
    # 分支条件：self.model == 'starcoder2'。
    # 分支条件：self.model == 'codellama7b'。
    # 分支条件：self.model == 'deepseekcoder'。
    # 分支条件：self.model == 'qwencoder'。
    # 分支条件：self.model == 'qwen3'。
    # 分支条件：self.model == 'llama3'。
    # 分支条件：self.model.startswith('gpt')。
    # 分支条件：self.model == 'gpt35'。
    # 分支条件：self.model == 'gpt4'。
    # 分支条件：self.model == 'gpt51'。
    # 显式抛出：ValueError(f'未知GPT模型标识: {self.model}')。
    # 显式抛出：ValueError(f'不支持的模型类型: {self.model}')。
    # 写入目标：self.tokenizer。
    # 写入目标：self.max_input_length。
    # 写入目标：os.environ['TIKTOKEN_CACHE_DIR']。
    # 写入目标：self.task_desc。
    # 直接调用：AutoTokenizer.from_pretrained。
    # 直接调用：self.model.startswith。
    # 直接调用：tiktoken.get_encoding。
    # 直接调用：ValueError。
    # [/maint-note:3910a2ca9a16]
    def _set_tokenizer(self):
        if self.model == "codegen":
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.codegen350m_repo)
            self.max_input_length = self.config.codegen_max_token - self.config.max_to_generate

        elif self.model == "codegen25":
            os.environ["TIKTOKEN_CACHE_DIR"] = self.config.tiktoken_cache_dir
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.codegen25_repo, trust_remote_code=True
            )
            self.max_input_length = self.config.codegen25_max_token - self.config.max_to_generate

        elif self.model == "santacoder":
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.santacoder_repo)
            self.max_input_length = self.config.santacoder_max_token - self.config.max_to_generate

        elif self.model == "starcoder":
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.starcoder_repo)
            self.max_input_length = self.config.starcoder_max_token - self.config.max_to_generate

        # =========================
        # StarCoder2
        # =========================
        elif self.model == "starcoder2":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.starcoder2_repo, trust_remote_code=True
            )
            self.max_input_length = self.config.starcoder2_max_token - self.config.max_to_generate

        elif self.model == "codellama7b":
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.codellama7b_repo)
            self.max_input_length = self.config.codellama_max_token - self.config.max_to_generate

        elif self.model == "deepseekcoder":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.deepseekcoder_repo, trust_remote_code=True
            )
            self.max_input_length = self.config.deepseek_max_token - self.config.max_to_generate

        # =========================
        # Qwen2.5-Coder-Instruct
        # =========================
        elif self.model == "qwencoder":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.qwencoder_repo, trust_remote_code=True
            )
            self.max_input_length = self.config.qwencoder_max_token - self.config.max_to_generate

        # =========================
        # 新增：Qwen3
        # =========================
        elif self.model == "qwen3":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.qwen3_repo, trust_remote_code=True
            )
            self.max_input_length = self.config.qwen3_max_token - self.config.max_to_generate

        # =========================
        # 新增：Llama3
        # =========================
        elif self.model == "llama3":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llama3_repo
            )
            self.max_input_length = self.config.llama3_max_token - self.config.max_to_generate

        elif self.model.startswith("gpt"):
            os.environ["TIKTOKEN_CACHE_DIR"] = self.config.tiktoken_cache_dir
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

            self.task_desc = (
                "You are a Java programming expert. "
                "Please complete the last line of the following Java code:\n"
            )

            if self.model == "gpt35":
                self.max_input_length = (
                    self.config.gpt35_max_token - self.config.max_to_generate - 16
                )
            elif self.model == "gpt4":
                self.max_input_length = (
                    self.config.gpt4_max_token - self.config.max_to_generate - 16
                )
            elif self.model == "gpt51":
                self.max_input_length = (
                    self.config.gpt51_max_token - self.config.max_to_generate - 16
                )
            else:
                raise ValueError(f"未知GPT模型标识: {self.model}")
        else:
            raise ValueError(f"不支持的模型类型: {self.model}")

    # [maint-note:d51f5512b3a6]
    # 维护导航：cal_token_nums；下列为本函数源码中的接口与路径索引。
    # 输入 text：类型由调用方与函数体约束。
    # 分支条件：self.model.startswith('codegen') or self.model == 'codellama7b' or self.model == 'deepseekcoder' or (self.model == 'qwencoder') or (self.model == 'qwen3') or (self.model == 'llama3')。
    # 分支条件：self.model.startswith('gpt')。
    # 返回路径：int(self.tokenizer(text, return_tensors='pt').attention_mask.sum().item())。
    # 返回路径：len(self.tokenizer.encode(text, disallowed_special=()))。
    # 返回路径：int(self.tokenizer.encode(text, return_tensors='pt').flatten().size(0))。
    # 直接调用：self.model.startswith。
    # 直接调用：int。
    # 直接调用：self.tokenizer(text, return_tensors='pt').attention_mask.sum().item。
    # 直接调用：self.tokenizer(text, return_tensors='pt').attention_mask.sum。
    # 直接调用：self.tokenizer。
    # 直接调用：len。
    # 直接调用：self.tokenizer.encode。
    # 直接调用：self.tokenizer.encode(text, return_tensors='pt').flatten().size。
    # 直接调用：self.tokenizer.encode(text, return_tensors='pt').flatten。
    # [/maint-note:d51f5512b3a6]
    def cal_token_nums(self, text):
        if (
            self.model.startswith("codegen")
            or self.model == "codellama7b"
            or self.model == "deepseekcoder"
            or self.model == "qwencoder"
            or self.model == "qwen3"
            or self.model == "llama3"
        ):
            return int(self.tokenizer(text, return_tensors="pt").attention_mask.sum().item())

        elif self.model.startswith("gpt"):
            return len(self.tokenizer.encode(text, disallowed_special=()))

        else:
            # santacoder / starcoder / starcoder2
            return int(self.tokenizer.encode(text, return_tensors="pt").flatten().size(0))

    # [maint-note:49e240412bb8]
    # 维护导航：cal_prompt_max_length；下列为本函数源码中的接口与路径索引。
    # 输入 program：类型由调用方与函数体约束。
    # 输入 suffix：类型由调用方与函数体约束。
    # 分支条件：program_len >= half_length。
    # 返回路径：half_length - suffix_len。
    # 返回路径：self.max_input_length - program_len - suffix_len。
    # 直接调用：self.cal_token_nums。
    # 直接调用：int。
    # [/maint-note:49e240412bb8]
    def cal_prompt_max_length(self, program, suffix):
        """
        Return the maximum length for prompt
        """
        suffix = "\n*/\n" + suffix

        suffix_len = self.cal_token_nums(suffix)
        program_len = self.cal_token_nums(program)

        half_length = int(0.5 * self.max_input_length)
        if program_len >= half_length:
            return half_length - suffix_len
        else:
            return self.max_input_length - program_len - suffix_len

    # [maint-note:40fc6f9dbd33]
    # 维护导航：judge_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 prompt：类型由调用方与函数体约束。
    # 输入 max_length：类型由调用方与函数体约束。
    # 分支条件：self.model.startswith('gpt')。
    # 返回路径：self.cal_token_nums(prompt) <= max_length。
    # 直接调用：self.model.startswith。
    # 直接调用：self.cal_token_nums。
    # [/maint-note:40fc6f9dbd33]
    def judge_prompt(self, prompt, max_length):
        """
        True: fine
        False: overlong
        """
        prompt = "/*\n" + prompt

        if self.model.startswith("gpt"):
            prompt = self.task_desc + prompt

        return self.cal_token_nums(prompt) <= max_length

    # [maint-note:999cb49722a4]
    # 维护导航：truncate_concat；下列为本函数源码中的接口与路径索引。
    # 输入 program：类型由调用方与函数体约束。
    # 输入 prompt：类型由调用方与函数体约束。
    # 输入 suffix：类型由调用方与函数体约束。
    # 分支条件：self.model.startswith('codegen') or self.model == 'codellama7b' or self.model == 'deepseekcoder' or (self.model == 'qwencoder') or (self.model == 'qwen3') or (self.model == 'llama3')。
    # 分支条件：self.model == 'santacoder' or self.model == 'starcoder' or self.model == 'starcoder2'。
    # 分支条件：self.model.startswith('gpt')。
    # 返回路径：truncated_prompt。
    # 直接调用：self.model.startswith。
    # 直接调用：self.codegen_truncate_concat。
    # 直接调用：self.coder_truncate_concat。
    # 直接调用：self.gpt_truncate_concat。
    # [/maint-note:999cb49722a4]
    def truncate_concat(self, program, prompt, suffix):
        truncated_prompt = None

        if (
            self.model.startswith("codegen")
            or self.model == "codellama7b"
            or self.model == "deepseekcoder"
            or self.model == "qwencoder"
            or self.model == "qwen3"
            or self.model == "llama3"
        ):
            truncated_prompt = self.codegen_truncate_concat(program, prompt, suffix)[0]

        elif (
            self.model == "santacoder"
            or self.model == "starcoder"
            or self.model == "starcoder2"
        ):
            truncated_prompt = self.coder_truncate_concat(program, prompt, suffix)[0]

        elif self.model.startswith("gpt"):
            truncated_prompt = self.gpt_truncate_concat(program, prompt, suffix)[0]

        return truncated_prompt

    # [maint-note:d3502d5e4cae]
    # 维护导航：codegen_truncate_concat；下列为本函数源码中的接口与路径索引。
    # 输入 program：类型由调用方与函数体约束。
    # 输入 prompt：类型由调用方与函数体约束。
    # 输入 suffix：类型由调用方与函数体约束。
    # 分支条件：program_len <= 0.5 * max_input_length。
    # 分支条件：prompt_wo_suffix_len > length4prompt。
    # 分支条件：prompt_len <= 0.5 * max_input_length。
    # 分支条件：program_len > length4program。
    # 返回路径：(prompts, input_cut_flag, prompt_cut_flag)。
    # 写入目标：self.tokenizer.truncation_side。
    # 直接调用：getattr。
    # 直接调用：self.tokenizer。
    # 直接调用：int。
    # 直接调用：program_token.attention_mask.sum().item。
    # 直接调用：program_token.attention_mask.sum。
    # 直接调用：prompt_token.attention_mask.sum().item。
    # 直接调用：prompt_token.attention_mask.sum。
    # 直接调用：suffix_token.attention_mask.sum().item。
    # 直接调用：suffix_token.attention_mask.sum。
    # 直接调用：torch.concat。
    # 直接调用：self.tokenizer.decode。
    # [/maint-note:d3502d5e4cae]
    def codegen_truncate_concat(self, program, prompt, suffix):
        input_cut_flag = False
        prompt_cut_flag = False

        max_input_length = self.max_input_length

        prefix = "/*\n"
        suffix = "\n*/\n" + suffix

        # ========== FIX: 使用 add_special_tokens=False 避免重复 BOS token ==========
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)

        # 对 suffix 和 program 编码时不添加特殊token
        suffix_token = self.tokenizer(suffix, return_tensors="pt", add_special_tokens=False)
        program_token = self.tokenizer(program, return_tensors="pt", add_special_tokens=False)

        # 对 prompt 编码时添加 BOS token（如果模型需要）
        prompt = prefix + prompt
        prompt_token = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        program_len = int(program_token.attention_mask.sum().item())
        prompt_wo_suffix_len = int(prompt_token.attention_mask.sum().item())
        suffix_len = int(suffix_token.attention_mask.sum().item())

        prompt_len = prompt_wo_suffix_len + suffix_len

        if program_len <= 0.5 * max_input_length:
            length4prompt = max_input_length - program_len - suffix_len

            if prompt_wo_suffix_len > length4prompt:
                self.tokenizer.truncation_side = "right"
                prompt_token = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=length4prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                prompt_cut_flag = True

        elif prompt_len <= 0.5 * max_input_length:
            length4program = max_input_length - prompt_len

            if program_len > length4program:
                self.tokenizer.truncation_side = "left"
                program_token = self.tokenizer(
                    program,
                    truncation=True,
                    max_length=length4program,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                input_cut_flag = True

        else:
            length4prompt = int(0.5 * max_input_length - suffix_len)
            length4program = int(0.5 * max_input_length)

            self.tokenizer.truncation_side = "right"
            prompt_token = self.tokenizer(
                prompt,
                truncation=True,
                max_length=length4prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )

            self.tokenizer.truncation_side = "left"
            program_token = self.tokenizer(
                program,
                truncation=True,
                max_length=length4program,
                return_tensors="pt",
                add_special_tokens=False,
            )

            prompt_cut_flag = True
            input_cut_flag = True

        concat_tokens = torch.concat(
            (
                prompt_token.input_ids[0],
                suffix_token.input_ids[0],
                program_token.input_ids[0],
            ),
            0,
        )

        prompts = self.tokenizer.decode(concat_tokens)
        return prompts, input_cut_flag, prompt_cut_flag

    # [maint-note:057cec0fef8e]
    # 维护导航：coder_truncate_concat；下列为本函数源码中的接口与路径索引。
    # 输入 program：类型由调用方与函数体约束。
    # 输入 prompt：类型由调用方与函数体约束。
    # 输入 suffix：类型由调用方与函数体约束。
    # 分支条件：program_len <= 0.5 * max_input_length。
    # 分支条件：prompt_wo_suffix_len > length4prompt。
    # 分支条件：prompt_len <= 0.5 * max_input_length。
    # 分支条件：program_len > length4program。
    # 返回路径：(prompts, input_cut_flag, prompt_cut_flag)。
    # 写入目标：self.tokenizer.truncation_side。
    # 直接调用：self.tokenizer.encode。
    # 直接调用：int。
    # 直接调用：program_token.flatten().size。
    # 直接调用：program_token.flatten。
    # 直接调用：suffix_token.flatten().size。
    # 直接调用：suffix_token.flatten。
    # 直接调用：prompt_token.flatten().size。
    # 直接调用：prompt_token.flatten。
    # 直接调用：torch.concat。
    # 直接调用：self.tokenizer.decode。
    # [/maint-note:057cec0fef8e]
    def coder_truncate_concat(self, program, prompt, suffix):
        input_cut_flag = False
        prompt_cut_flag = False

        max_input_length = self.max_input_length

        prefix = "/*\n"
        suffix = "\n*/\n" + suffix

        suffix_token = self.tokenizer.encode(suffix, return_tensors="pt")
        program_token = self.tokenizer.encode(program, return_tensors="pt")

        prompt = prefix + prompt
        prompt_token = self.tokenizer.encode(prompt, return_tensors="pt")

        program_len = int(program_token.flatten().size(0))
        suffix_len = int(suffix_token.flatten().size(0))
        prompt_wo_suffix_len = int(prompt_token.flatten().size(0))

        prompt_len = prompt_wo_suffix_len + suffix_len

        if program_len <= 0.5 * max_input_length:
            length4prompt = max_input_length - program_len - suffix_len

            if prompt_wo_suffix_len > length4prompt:
                self.tokenizer.truncation_side = "right"
                prompt_token = self.tokenizer.encode(
                    prompt,
                    truncation=True,
                    max_length=length4prompt,
                    return_tensors="pt",
                )
                prompt_cut_flag = True

        elif prompt_len <= 0.5 * max_input_length:
            length4program = max_input_length - prompt_len

            if program_len > length4program:
                self.tokenizer.truncation_side = "left"
                program_token = self.tokenizer.encode(
                    program,
                    truncation=True,
                    max_length=length4program,
                    return_tensors="pt",
                )
                input_cut_flag = True

        else:
            length4prompt = int(0.5 * max_input_length - suffix_len)
            length4program = int(0.5 * max_input_length)

            self.tokenizer.truncation_side = "right"
            prompt_token = self.tokenizer.encode(
                prompt,
                truncation=True,
                max_length=length4prompt,
                return_tensors="pt",
            )

            self.tokenizer.truncation_side = "left"
            program_token = self.tokenizer.encode(
                program,
                truncation=True,
                max_length=length4program,
                return_tensors="pt",
            )

            prompt_cut_flag = True
            input_cut_flag = True

        concat_tokens = torch.concat(
            (prompt_token[0], suffix_token[0], program_token[0]), 0
        )

        prompts = self.tokenizer.decode(concat_tokens)
        return prompts, input_cut_flag, prompt_cut_flag

    # [maint-note:f2288c651624]
    # 维护导航：gpt_truncate_concat；下列为本函数源码中的接口与路径索引。
    # 输入 program：类型由调用方与函数体约束。
    # 输入 prompt：类型由调用方与函数体约束。
    # 输入 suffix：类型由调用方与函数体约束。
    # 分支条件：prompt is None and suffix is None。
    # 分支条件：program_len > max_program_length。
    # 分支条件：program_len <= 0.5 * max_input_length。
    # 分支条件：prompt_wo_suffix_len > length4prompt。
    # 分支条件：prompt_len <= 0.5 * max_input_length。
    # 分支条件：program_len > length4program。
    # 返回路径：(self.task_desc + program, input_cut_flag, prompt_cut_flag)。
    # 返回路径：(concat_message, input_cut_flag, prompt_cut_flag)。
    # 直接调用：len。
    # 直接调用：self.tokenizer.encode。
    # 直接调用：self.tokenizer.decode。
    # 直接调用：int。
    # [/maint-note:f2288c651624]
    def gpt_truncate_concat(self, program, prompt, suffix):
        input_cut_flag = False
        prompt_cut_flag = False

        if prompt is None and suffix is None:
            desc_length = len(
                self.tokenizer.encode(self.task_desc, disallowed_special=())
            )
            max_program_length = self.max_input_length - desc_length

            program_token = self.tokenizer.encode(program, disallowed_special=())
            program_len = len(program_token)
            if program_len > max_program_length:
                program_token = program_token[-max_program_length:]
                input_cut_flag = True
                program = self.tokenizer.decode(program_token)

            return self.task_desc + program, input_cut_flag, prompt_cut_flag

        max_input_length = self.max_input_length

        prefix = self.task_desc + "/*\n"
        suffix = "\n*/\n" + suffix

        suffix_token = self.tokenizer.encode(suffix, disallowed_special=())
        program_token = self.tokenizer.encode(program, disallowed_special=())

        prompt = prefix + prompt
        prompt_token = self.tokenizer.encode(prompt, disallowed_special=())

        program_len = len(program_token)
        prompt_wo_suffix_len = len(prompt_token)
        suffix_len = len(suffix_token)

        prompt_len = prompt_wo_suffix_len + suffix_len

        if program_len <= 0.5 * max_input_length:
            length4prompt = max_input_length - program_len - suffix_len

            if prompt_wo_suffix_len > length4prompt:
                prompt_token = prompt_token[:length4prompt]
                prompt_cut_flag = True
                prompt = self.tokenizer.decode(prompt_token)

        elif prompt_len <= 0.5 * max_input_length:
            length4program = max_input_length - prompt_len

            if program_len > length4program:
                program_token = program_token[-length4program:]
                input_cut_flag = True
                program = self.tokenizer.decode(program_token)

        else:
            length4prompt = int(0.5 * max_input_length - suffix_len)
            length4program = int(0.5 * max_input_length)

            prompt_token = prompt_token[:length4prompt]
            prompt = self.tokenizer.decode(prompt_token)
            program_token = program_token[-length4program:]
            program = self.tokenizer.decode(program_token)

            prompt_cut_flag = True
            input_cut_flag = True

        concat_message = prompt + suffix + program
        return concat_message, input_cut_flag, prompt_cut_flag
