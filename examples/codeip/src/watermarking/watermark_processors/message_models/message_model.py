from typing import Union, Optional
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
import torch

from .base_message_model import BaseMessageModelFast, TextTooShortError
from ...utils.hash_fn import Hash1

HfTokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]


class RandomMessageModel(BaseMessageModelFast):
    def __init__(self, tokenizer: HfTokenizer, lm_tokenizer: HfTokenizer,
                 delta, seed=42, message_code_len=10, hash_prefix_len=1,
                 hash_fn=Hash1, device='cuda:0'):
        super().__init__(seed, message_code_len)
        self.tokenizer = tokenizer
        self.lm_tokenizer = lm_tokenizer
        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        self.delta = delta
        self.message_code_len = message_code_len
        self.hash_prefix_len = hash_prefix_len
        self.hash_fn = hash_fn
        self.device = device
        self.tokenizer_id_2_lm_tokenizer_id_list = self.construct_tokenizer_id_2_lm_tokenizer_id_list()
        self.max_vocab_size = 100000

    @property
    def random_delta(self):
        return self.delta * 1.

    def set_random_state(self):
        pass

    def construct_tokenizer_id_2_lm_tokenizer_id_list(self):
        tokenizer_id_2_lm_tokenizer_id_list = torch.zeros(len(self.tokenizer.vocab),
                                                          dtype=torch.long)
        for v, i in self.tokenizer.vocab.items():
            tokenizer_id_2_lm_tokenizer_id_list[i] = self.lm_tokenizer.convert_tokens_to_ids(v)
        return tokenizer_id_2_lm_tokenizer_id_list.to(self.device)

    def check_x_prefix_x_cur_messages(self, x_prefix: Optional[torch.Tensor] = None,
                                      x_cur: Optional[torch.Tensor] = None,
                                      messages: Optional[torch.Tensor] = None):
        if x_prefix is not None:
            assert x_prefix.dim() == 2
        if x_cur is not None:
            assert x_cur.dim() == 2
        if messages is not None:
            assert messages.dim() == 1
        if x_prefix is not None and x_cur is not None:
            assert x_prefix.shape[0] == x_cur.shape[0]

    def get_x_prefix_seed_element(self, x_prefix, transform: bool):
        self.check_x_prefix_x_cur_messages(x_prefix=x_prefix)
        if x_prefix.shape[1] < self.hash_prefix_len:
            raise TextTooShortError
        x_prefix = x_prefix[:, -self.hash_prefix_len:]
        if transform:
            x_prefix = self.tokenizer_id_2_lm_tokenizer_id_list[x_prefix]
        seed_elements = [tuple(_.tolist()) for _ in x_prefix]
        # print(seed_elements)
        return seed_elements

    def get_x_prefix_seeds(self, x_prefix, transform: bool):
        '''
        :param x_prefix: [batch_size, seq_len]

        :return: seeds: list of [batch_size]
        '''
        self.check_x_prefix_x_cur_messages(x_prefix=x_prefix)
        seed_elements = self.get_x_prefix_seed_element(x_prefix, transform=transform)
        seeds = torch.tensor([hash(seed_element + (self.seed,)) for seed_element in seed_elements],
                             device=x_prefix.device)
        return seeds

    def get_x_prefix_and_message_seeds(self, x_prefix: torch.Tensor, messages: torch.Tensor,
                                       x_prefix_seeds: torch.Tensor):
        '''
        :param x_prefix: [batch_size, seq_len]
        :param message_or_messages: [message_num]

        :return: [batch_size, message_num]
        '''
        seeds = (x_prefix_seeds % (2 ** (32 - self.message_code_len))).unsqueeze(1)
        seeds = seeds * (2 ** self.message_code_len) + messages.unsqueeze(0)
        seeds = self.hash_fn(seeds)
        return seeds

    def get_x_prefix_and_message_and_x_cur_seeds(self, x_prefix: torch.Tensor,
                                                 messages: torch.Tensor, x_cur: torch.Tensor,
                                                 x_prefix_seeds: torch.Tensor,
                                                 x_prefix_and_message_seeds: torch.Tensor = None):
        '''
        :param x_prefix: [batch_size, seq_len]
        :param message_or_messages: [message_num]
        :param x_cur: [batch_size, vocab_size]

        return: [batch_size, message_num, vocab_size]
        '''
        assert x_cur.dim() == 2 and x_prefix.dim() == 2
        if x_prefix_and_message_seeds is None:
            x_prefix_and_message_seeds = self.get_x_prefix_and_message_seeds(x_prefix, messages,
                                                                             x_prefix_seeds)
        x_prefix_and_message_and_x_cur_seeds = self.hash_fn(x_prefix_and_message_seeds.unsqueeze(
            -1) * self.max_vocab_size + x_cur.unsqueeze(1))
        return x_prefix_and_message_and_x_cur_seeds

    def cal_log_P_with_single_message(self, x_prefix: torch.LongTensor, x_cur: torch.Tensor,
                                      messages: torch.Tensor,
                                      transform_x_cur: bool = True) -> torch.Tensor:
        '''
        :param x_prefix: [batch, seq_len]
        :param x_cur: [batch, vocab_size]
        :param messages: [message_num]
        :param lm_predictions: [batch, all_vocab_size]

        :return: log of P of size [batch, messages, vocab_size]
        '''
        assert messages.numel() == 1
        self.check_x_prefix_x_cur_messages(x_prefix=x_prefix, x_cur=x_cur, messages=messages)

        if transform_x_cur:
            x_cur = self.tokenizer_id_2_lm_tokenizer_id_list[x_cur]

        x_prefix_seeds = self.get_x_prefix_seeds(x_prefix, transform=True)

        x_prefix_and_message_and_x_cur_seeds = self.get_x_prefix_and_message_and_x_cur_seeds(
            x_prefix, messages, x_cur, x_prefix_seeds=x_prefix_seeds)
        x_prefix_and_message_and_x_cur_idxs = x_prefix_and_message_and_x_cur_seeds % 2

        log_Ps = x_prefix_and_message_and_x_cur_idxs.float() * self.delta

        return log_Ps

    def cal_log_P_with_single_x_cur(self, x_prefix: torch.LongTensor, x_cur: torch.Tensor,
                                    messages: torch.Tensor):
        '''
        :param x_prefix: [batch, seq_len]
        :param x_cur: [batch, vocab_size]
        :param messages: [message_num]
        :param lm_predictions: [batch, all_vocab_size]

        :return: log of P of size [batch, messages, vocab_size]
        '''
        assert x_cur.shape[1] == 1

        x_prefix_seeds = self.get_x_prefix_seeds(x_prefix, transform=False)

        x_prefix_and_message_and_x_cur_seeds = self.get_x_prefix_and_message_and_x_cur_seeds(
            x_prefix, messages, x_cur, x_prefix_seeds=x_prefix_seeds)
        x_prefix_and_message_and_x_cur_idxs = x_prefix_and_message_and_x_cur_seeds % 2
        log_Ps = x_prefix_and_message_and_x_cur_idxs.float() * self.delta

        return log_Ps

    def cal_log_Ps(self, x_prefix: torch.LongTensor, x_cur: torch.Tensor,
                   messages: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        '''
        :param x_prefix: [batch, seq_len]
        :param x_cur: [batch, vocab_size]
        :param messages: [message_num]
        :param lm_predictions: [batch, all_vocab_size]

        :return: log of P of size [batch, messages, vocab_size]
        '''

        # check input shape
        self.check_x_prefix_x_cur_messages(x_prefix, x_cur, messages)

        if messages.numel() == 1:
            # cal P(x_prefix, x_cur, message) for all possible x_cur
            log_Ps = []
            for message in messages:
                message = message.reshape(-1)
                log_Ps_single_message = self.cal_log_P_with_single_message(x_prefix, x_cur,
                                                                           message)
                log_Ps.append(log_Ps_single_message)
            log_Ps = torch.stack(log_Ps, dim=1)
        elif x_cur.shape[1] == 1:
            log_Ps = self.cal_log_P_with_single_x_cur(x_prefix, x_cur, messages)
        else:
            raise NotImplementedError

        return log_Ps
