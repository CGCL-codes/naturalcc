from typing import Union
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from .base_processor import WmProcessorBase
from .message_models.message_model import RandomMessageModel, TextTooShortError

HfTokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]


def convert_input_ids_to_key(input_ids):
    if len(input_ids.shape) == 2:
        assert input_ids.shape[0] == 1
        input_ids = input_ids[0]
    return tuple(input_ids.tolist())


class WmProcessorRandomMessageModel(WmProcessorBase):
    def __init__(self, message, message_model: RandomMessageModel, tokenizer, encode_ratio=10.,
                 seed=42, strategy='vanilla', max_confidence_lbd=0.5, top_k=1000):
        super().__init__(seed=seed)
        self.message = message
        if not isinstance(self.message, torch.Tensor):
            self.message = torch.tensor(self.message)
        self.message_model: RandomMessageModel = message_model
        self.message_code_len = self.message_model.message_code_len
        self.encode_ratio = encode_ratio
        self.encode_len = int(self.message_code_len * self.encode_ratio)
        self.start_length = 0
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.lm_prefix_len = self.message_model.hash_prefix_len
        if hasattr(self.message_model, 'tokenizer'):
            assert self.message_model.tokenizer == tokenizer
        self.strategy = strategy
        if self.strategy in ['max_confidence', 'max_confidence_updated']:
            raise NotImplementedError

    def set_random_state(self):
        pass

    @property
    def decode_messages(self):
        if not hasattr(self, '_decode_messages'):
            self._decode_messages = torch.arange(2 ** self.message_code_len,
                                                 device=self.message_model.device)
        return self._decode_messages

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.message.device != input_ids.device:
            self.message = self.message.to(input_ids.device)
        try:
            input_ids = input_ids[:, self.start_length:]
            if input_ids.shape[1] < self.lm_prefix_len:
                raise TextTooShortError
            message_len = input_ids.shape[1] - self.lm_prefix_len
            cur_message = self.message[message_len // self.encode_len]
            cur_message = cur_message.reshape(-1)
            topk_indices = torch.topk(scores, self.top_k, dim=1)[1]
            log_Ps = self.message_model.cal_log_Ps(input_ids, x_cur=topk_indices,
                                                   messages=cur_message)
            log_Ps = log_Ps.squeeze()
            log_Ps = log_Ps.unsqueeze(0)
            scores.scatter_(1, topk_indices, log_Ps, reduce='add')
        except TextTooShortError:
            pass
        return scores

    def decode_with_input_ids(self, input_ids, messages = None, batch_size=16, disable_tqdm=False,
                              non_analyze=False):
        if messages is None:
            messages = self.decode_messages
        else:
            if not isinstance(messages, torch.Tensor):
                messages = torch.tensor(messages, device=input_ids.device)
        all_log_Ps = []
        assert input_ids.shape[0] == 1
        for i in tqdm(range(0, input_ids.shape[1] - self.lm_prefix_len - 1, batch_size),
                      disable=disable_tqdm):
            batch_input_ids = []
            for j in range(i, min(i + batch_size, input_ids.shape[1] - self.lm_prefix_len - 1)):
                batch_input_ids.append(input_ids[:, j:j + self.lm_prefix_len + 1])
            batch_input_ids = torch.cat(batch_input_ids, dim=0)
            x_prefix = batch_input_ids[:, :-1]
            x_cur = batch_input_ids[:, -1:]
            log_Ps = self.message_model.cal_log_Ps(x_prefix, x_cur, messages=messages)
            all_log_Ps.append(log_Ps)
        all_log_Ps = torch.cat(all_log_Ps, dim=0)
        all_log_Ps = all_log_Ps.squeeze()
        decoded_messages = []
        decoded_confidences = []
        for i in range(0, all_log_Ps.shape[0], self.encode_len):
            if not non_analyze:
                nums = (all_log_Ps[i:i + self.encode_len] > 0).sum(0)
                max_values, max_indices = torch.max(nums, 0)
                top_values, top_indices = torch.topk(nums, 2)
                decoded_message = messages[max_indices]
                decoded_confidence = int(max_values)
                decoded_probs = float(torch.softmax(nums.float(), dim=-1)[max_indices])
                relative_decoded_confidence = int(max_values) - int(top_values[1])
                decoded_messages.append(decoded_message)
                decoded_confidences.append(
                    (decoded_confidence, relative_decoded_confidence, decoded_probs))
            else:
                nums = (all_log_Ps[i:i + self.encode_len] > 0).sum(0)
                decoded_probs = torch.softmax(nums.float(), dim=-1)
                decoded_message, decoded_prob = decoded_probs.max(0)
                decoded_message = int(decoded_message)
                decoded_messages.append(decoded_message)
                decoded_prob = float(decoded_prob)
                decoded_confidences.append(
                    (-1, -1, decoded_prob))
        decoded_messages = [int(_) for _ in decoded_messages]
        if non_analyze:
            return decoded_messages, (None, decoded_confidences)
        else:
            return decoded_messages, (all_log_Ps.cpu(), decoded_confidences)

    def decode(self, text, messages=None, batch_size=16, disable_tqdm=False, non_analyze=False):
        input_ids = self.message_model.lm_tokenizer(text, return_tensors='pt')['input_ids'].to(
            self.message_model.device)
        return self.decode_with_input_ids(input_ids, messages, batch_size, disable_tqdm
                                          , non_analyze=non_analyze)
