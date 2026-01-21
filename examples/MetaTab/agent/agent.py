import os
import openai
from typing import Optional
from agent.model import Model
from utils.execute import markdown_to_df, parse_code_from_string, python_repl_ast, print_partial_markdown

# global variables for python repl
import pandas as pd
import numpy as np
from datetime import datetime


class TableAgent:
    def __init__(self, 
                 table: pd.DataFrame | str,
                 prompt_type: str,
                 model: Optional[Model],
                 long_model: Optional[Model],
                 temperature: float = 0.8,
                 top_p: float = 0.95,
                 stop_tokens: Optional[list] = ["Observation:"],
                 max_depth: int = 5,
                 log_dir: Optional[str] = None,
                 print_process: bool = False,
                 use_full_table: bool = True
                ):
        
        # if table is dataframe
        if isinstance(table, pd.DataFrame):
            self.df = table
        # if table is markdown string
        elif isinstance(table, str):
            self.df = markdown_to_df(table)
            
        self.model = model
        self.long_model = long_model
        self.max_depth = max_depth
        self.stop_tokens = stop_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.log_dir = log_dir
        self.use_full_table = use_full_table
        self.print_process = print_process
        
        if prompt_type == "wtq":
            from prompt.wtq.agent import agent_prefix, agent_prefix_with_omitted_rows_guideline
        elif prompt_type == "tabfact":
            from prompt.tabfact.agent import agent_prefix, agent_prefix_with_omitted_rows_guideline
        
        self.agent_prefix = agent_prefix
        self.agent_prefix_with_omitted_rows_guideline = agent_prefix_with_omitted_rows_guideline

        if self.use_full_table:
            table = self.df.to_markdown()
            self.prompt = agent_prefix
        else:
            table = print_partial_markdown(self.df)
            self.prompt = agent_prefix_with_omitted_rows_guideline


        self.prompt = self.prompt.replace("[TABLE]", table)
            
    
    def reset_prompt(self):
        if self.use_full_table:
            table = self.df.to_markdown()
            self.prompt = self.agent_prefix
            self.prompt = self.prompt.replace("[TABLE]", table)
        else:
            table = print_partial_markdown(self.df)
            self.prompt = self.agent_prefix_with_omitted_rows_guideline
            self.prompt = self.prompt.replace("[TABLE]", table)
    
    def query(self, temperature: Optional[float] = None) -> str:
        # encode the prompt to get the length of the prompt
        prompt_length = len(self.long_model.tokenizer.encode(self.prompt))

        if isinstance(self.model, Model):
            if prompt_length <= 3328:
                text, response = self.model.query(
                    prompt=self.prompt,
                    temperature=self.temperature if temperature is None else temperature,
                    top_p=self.top_p,
                    max_tokens= 4000 - prompt_length,
                    stop=self.stop_tokens
                )
            elif prompt_length <= 14592:
                print(f"Prompt length -- {prompt_length} is too long, we use the 16k version.")
                text, response = self.long_model.query(
                    prompt=self.prompt,
                    temperature=self.temperature if temperature is None else temperature,
                    top_p=self.top_p,
                    max_tokens= 15360 - prompt_length,
                    stop=self.stop_tokens
                )
            else:
                print(f"Prompt length -- {prompt_length} is too long, we cannot query the API.")
                text, response = "PROMPT TOO LONG, WE CAN NOT QUERY THE API", None
        
        else:
            # simply query the long model
            text, response = self.long_model.query(
                prompt=self.prompt,
                temperature=self.temperature if temperature is None else temperature,
                top_p=self.top_p,
                max_tokens= 15360 - prompt_length,
                stop=self.stop_tokens
            )
    
        return text, response

    def is_terminal(self, text: str) -> bool:

        return "Final Answer: " in text or "answer_directly" in text or "PROMPT TOO LONG, WE CAN NOT QUERY THE API" in text
    
    # dummy run for debugging
    def dummy_run(self, question:str, title:str) -> str:
        # reset the prompt
        self.reset_prompt()

        # construct the prompt
        self.prompt = self.prompt.replace("[TITLE]", title).replace("[QUESTION]", question).strip()

        # dummy text
        dummy_text = [
            "Action: python_repl_ast\nAction Input: `df.columns`\n",
            "Action: `python_repl_ast`\nAction Input: `df.iloc[0]`\n",
            "Action: `python_repl_ast`\nAction Input: ```python\ndf.columns[0]\n```\n",
            "Action: python_repl_ast\nAction Input: ```python\nthis will cause an error\n```\n",
            "Action: python_repl_ast\nAction Input: `df['Deaths Outside of Prisons & Camps']`"
        ]

        response_text = ""
        response_list = []
        new_line = "\n"
        memory = {}
        for i in range(self.max_depth):
            # mimic the response, we don't need to query the API
            text, response = dummy_text[i], None

            if self.is_terminal(text):
                break
            # get how many new lines in the text
            if i == 0:
                if "\n\n" in text:
                    new_line = "\n\n"
            else:
                text = new_line + text

            response_text += text
            response_list.append(response)

            # get the code from the response
            if "Action Input:" in text:
                code = parse_code_from_string(text.split("Action Input:")[-1].strip("\n").strip())
            elif "Action:" in text:
                code = parse_code_from_string(text.split("Action:")[-1].strip("\n").strip())
            else:
                code = parse_code_from_string(text)

            print(f"Run the code below:\n```\n{code}\n```")

            # execute the code
            observation, memory = python_repl_ast(code, custom_locals={"df": self.df}, custom_globals=globals(), memory=memory)
            
            print(f"Observation:\n```\n{observation}\n```")

            if isinstance(observation, str) and observation == "":
                observation = "success!"

            # if observation has multiple lines, we need to add new line at the beginning
            if "\n" in str(observation):
                observation = "\n" + str(observation)

            response_text += f"Observation: {observation}"
            self.prompt += text + f"Observation: {observation}"

        
        return response_text, response_list

    def run(self, question:str, title:str) -> str:
        # reset the prompt
        self.reset_prompt()

        # construct the prompt
        self.prompt = self.prompt.replace("[TITLE]", title).replace("[QUESTION]", question).strip()

        if self.log_dir is not None:
            with open(self.log_dir, "a") as f:
                f.write("=" *50 + "\n")
                f.write(self.prompt + "\n")
        
        if self.print_process:
            print("=" * 50)
            print(self.prompt)

        response_text = ""
        response_list = []
        new_line = "\n"
        memory = {}
        for i in range(self.max_depth):
            text, response = self.query()

            # get how many new lines in the text
            if i == 0:
                if "\n\n" in text:
                    new_line = "\n\n"
            else:
                text = new_line + text


            response_text += text
            response_list.append(response)
            
            if self.print_process:
                # print without new line
                print(text, end="")

            # first check if it is terminal
            if self.is_terminal(text):
                break
        
            # get the code from the response
            if "Action Input:" in text:
                code = parse_code_from_string(text.split("Action Input:")[-1].strip("\n").strip())
            elif "Action:" in text:
                code = parse_code_from_string(text.split("Action:")[-1].strip("\n").strip())
            else:
                code = parse_code_from_string(text)

            # execute the code, we need to pass the dataframe, and pandas as pd, numpy as np to the locals
            observation, memory = python_repl_ast(code, custom_locals={"df": self.df}, custom_globals=globals(), memory=memory)

            if isinstance(observation, str) and observation == "":
                observation = "success!"

            # if observation has multiple lines, we need to add new line at the beginning
            if "\n" in str(observation):
                observation = "\n" + str(observation)

            response_text += f"Observation: {observation}"
            self.prompt += text + f"Observation: {observation}"
            
            if self.print_process:
                print(f"Observation: {observation}", end="")
                
        
        # run out of depth, no terminal state, we still need to log the response
        if self.log_dir is not None:
            with open(self.log_dir, "a") as f:
                f.write(response_text + "\n")

        
        return response_text, response_list
