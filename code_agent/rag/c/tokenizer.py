#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml

from transformers import AutoTokenizer
import torch
import tiktoken
import attridict


class CModelTokenizer:
    def __init__(self, model):
        self.model = model
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r", encoding="utf-8") as config_file:
            self.config = attridict(
                yaml.load(config_file, Loader=yaml.FullLoader)
            )
        self._set_tokenizer()

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
                "You are a C programming expert. "
                "Please complete the last line of the following C code:\n"
            )

            if self.model == "gpt35":
                self.max_input_length = (
                    self.config.gpt35_max_token - self.config.max_to_generate - 16
                )
            elif self.model == "gpt4":
                self.max_input_length = (
                    self.config.gpt4_max_token - self.config.max_to_generate - 16
                )
            else:
                raise ValueError(f"未知GPT模型标识: {self.model}")
        else:
            raise ValueError(f"不支持的模型类型: {self.model}")

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

    def judge_prompt(self, prompt, max_length):
        """
        True: fine
        False: overlong
        """
        prompt = "/*\n" + prompt

        if self.model.startswith("gpt"):
            prompt = self.task_desc + prompt

        return self.cal_token_nums(prompt) <= max_length

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
