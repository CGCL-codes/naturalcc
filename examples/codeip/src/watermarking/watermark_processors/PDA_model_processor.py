from typing import Union

import torch
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from .base_processor import WmProcessorBase
from .message_models.PDA_message_model import PDAMessageModel

HfTokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]

class PDAProcessorMessageModel(WmProcessorBase):
    def __init__(self, message_model: PDAMessageModel, tokenizer, gamma, encode_ratio=10.,
                 seed=42):
        super().__init__(seed=seed)
        
      
        self.message_model: PDAMessageModel = message_model
        
        self.encode_ratio = encode_ratio
        self.start_length = 0
        self.tokenizer = tokenizer
        self.gamma = gamma


    def set_random_state(self):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = input_ids[:, self.start_length:]

        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        predicted_class = self.message_model.get_pda_predictions(input_texts)
        
        scores = self.message_model.cal_addition_scores(input_ids,
                                                lm_predictions=predicted_class,
                                                scores=scores, 
                                                gamma=self.gamma)
        return scores
