from typing import Union
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
import torch

from ...utils.hash_fn import Hash1
from ...arg_classes.wm_arg_class import WmBaseArgs

HfTokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]

import torch.nn.functional as F
from pygments.lexers import PythonLexer, GoLexer, JavaLexer, JavascriptLexer, PhpLexer
from pygments.token import  STANDARD_TYPES


class PDAMessageModel():
    def __init__(self, tokenizer: HfTokenizer, pda_model,
                 delta, seed=42, lm_topk=1000, message_code_len=10,
                 random_permutation_num=100, hash_prefix_len=1, hash_fn=Hash1):
        self.tokenizer = tokenizer
        self.delta = delta
        self.message_code_len = message_code_len
        if WmBaseArgs.language == "python":
            self.lexer = PythonLexer()
        elif WmBaseArgs.language == "go":
            self.lexer = GoLexer()
        elif WmBaseArgs.language == "java":
            self.lexer = JavaLexer()
        elif WmBaseArgs.language == "javascript":
            self.lexer = JavascriptLexer()
        elif WmBaseArgs.language == "php":
            self.lexer = PhpLexer()
        self.nonterminal2id = {nonterminal: i for i, nonterminal in enumerate(STANDARD_TYPES.values())}
        self.lex_and_tokenizer_id_list = self.construct_lex_and_tokenizer_id_list()
        self.max_vocab_size = 100000
        self.pda_model = pda_model


    @property
    def random_delta(self):
        return self.delta * 1.

    @property
    def device(self):
        print("not implemented")

    def set_random_state(self):
        pass
    
    def construct_lex_and_tokenizer_id_list(self):
        lex_and_tokenizer_id_list = [set() for i in range(len(self.nonterminal2id))]
        for i in range(self.tokenizer.vocab_size):
            v = self.tokenizer.decode([i])
            tokens = self.lexer.get_tokens(v)
            for token in tokens:
                token_type, value = token
                lex_and_tokenizer_id_list[self.nonterminal2id[STANDARD_TYPES[token_type]]].add(i)

        return lex_and_tokenizer_id_list
    
    def get_pda_predictions(self, strings):
        tokens = self.lexer.get_tokens(strings[0])
        lex_array = [self.nonterminal2id[STANDARD_TYPES[token_type]] for token_type, value in tokens]
        lex_array = torch.tensor([lex_array]).to(WmBaseArgs.device)
        score = self.pda_model(lex_array)
        output_probs = F.log_softmax(score, dim=1)
        predicted_class = torch.argmax(output_probs, dim=1)      
        return predicted_class
        


    def cal_addition_scores(self, x_prefix: torch.LongTensor, 
                   lm_predictions, scores: torch.Tensor, gamma
                   ):
        selected_idx = list(self.lex_and_tokenizer_id_list[lm_predictions])
        scores[:, selected_idx] += gamma
        return scores
