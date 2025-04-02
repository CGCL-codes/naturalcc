import transformers
import torch
from torch import nn
import openai
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from tqdm import tqdm
import datasets
import json
import argparse
import os
# device = "cuda"


class FeatureExtractor(nn.Module):
    def __init__(self, model_card:str):
        super(FeatureExtractor, self).__init__()

        if(model_card.split('/')[0]=='openai'):
            print("Load OpenAI model")
            self.openai = True
        else:
            self.openai = False

        if(self.openai):
            self.model_card = model_card.split('/')[1]
            self.model = ""
            return

        def GetConfig():
            return transformers.AutoConfig.from_pretrained(self.model_card)

        def GetTokenizer():
            if (isinstance(self.config, transformers.GPT2Config)):
                tokenizer = transformers.GPT2Tokenizer.from_pretrained(self.model_card)
            elif (isinstance(self.config, transformers.RobertaConfig)):
                tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_card)
            elif (isinstance(self.config, transformers.T5Config)):
                tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_card)
            elif (isinstance(self.config, transformers.PLBartConfig)):
                tokenizer = transformers.PLBartTokenizer.from_pretrained(self.model_card)
            elif (isinstance(self.config, transformers.CodeGenConfig)):
                tokenizer = transformers.CodeGenTokenizer.from_pretrained(self.model_card)
            else:
                assert False

                # Special Tokens
            if (tokenizer.pad_token_id == None):
                tokenizer.add_special_tokens({'pad_token': '<pad>'})

            tokenizer.model_max_length = 512
            # tokenizer.model_max_length = 128
            return tokenizer

        def GetModel():
            if (isinstance(self.config, transformers.GPT2Config)):
                return transformers.GPT2Model.from_pretrained(self.model_card)
            elif (isinstance(self.config, transformers.RobertaConfig)):
                return transformers.RobertaModel.from_pretrained(self.model_card)
            elif (isinstance(self.config, transformers.T5Config)):
                return transformers.T5EncoderModel.from_pretrained(self.model_card)
            elif (isinstance(self.config, transformers.PLBartConfig)):
                return transformers.PLBartModel.from_pretrained(self.model_card)
            # elif(isinstance(self.config, transformers.XGLMConfig)):
            # return transformers.XGLMModel.from_pretrained(self.model_card)
            elif (isinstance(self.config, transformers.CodeGenConfig)):
                return transformers.CodeGenModel.from_pretrained(self.model_card)

            else:
                assert False

        self.model_card = model_card
        self.config = GetConfig()
        self.tokenizer = GetTokenizer()
        self.model = GetModel()
        # if torch.cuda.device_count() > 1:
        #     # multi-gpu
        #     self.model = torch.nn.DataParallel(self.model)

    def openai_forward(self, input_seq):
        openai.api_key = "sk-XXXXXXX"
        input_seq = [' '.join(i.split(' ')[:min(512, len(i)-1)]) for i in input_seq]
        # print(input_seq[0])
        # raise NotImplementedError("fuck")
        response = openai.Embedding.create(
            input=input_seq,
            engine=self.model_card)
        return [i["embedding"] for i in response["data"]]
        # pass

    def Tokenize(self, input_strs: list):
        return self.tokenizer(input_strs, padding="max_length", truncation=True, return_tensors="pt")

    def forward(self, input_ids, attention_mask):
        # tok = self.tokenizer(input_str, return_tensors="pt")
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        # GPT / CODEGEN
        if (isinstance(self.config, transformers.GPT2Config) or
                isinstance(self.config, transformers.CodeGenConfig)):
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1

            return outputs.last_hidden_state[:, sequence_lengths, :]
        # ROBERTA
        elif (isinstance(self.config, transformers.RobertaConfig)):
            # return torch.mean(outputs.last_hidden_state,1)
            return outputs.last_hidden_state[:, 0, :]
        # T5
        elif (isinstance(self.config, transformers.T5Config)):
            hidden_states = outputs['last_hidden_state']
            eos_mask = input_ids.eq(self.config.eos_token_id)

            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                  hidden_states.size(-1))[:, -1, :]
            return vec
        # PLBART
        elif (isinstance(self.config, transformers.PLBartConfig)):
            eos_mask = input_ids.eq(self.config.eos_token_id)
            hidden_states = outputs[0]
            sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                                      hidden_states.size(-1))[
                                      :, -1, :
                                      ]
            return sentence_representation
        # XGLM (deprecated)
        # elif(isinstance(self.config, transformers.XGLMConfig)):
        # return torch.mean(outputs.last_hidden_state,1)
        else:
            assert False

    def GetDimension(self):
        dim = -1
        with torch.no_grad():
            dim = self.forward(**self.Tokenize(["dummy"])).shape[-1]
        return dim

