import json
import os.path

import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, NoRepeatNGramLogitsProcessor, RepetitionPenaltyLogitsProcessor
from .arg_classes.wm_arg_class import WmBaseArgs
from .watermark_processors.message_model_processor import WmProcessorRandomMessageModel
from .watermark_processors.message_models.message_model import RandomMessageModel

from .watermark_processors.PDA_model_processor import PDAProcessorMessageModel
from .watermark_processors.message_models.PDA_message_model import PDAMessageModel
import torch.nn as nn

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def truncate(d, max_length=200):
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and len(v.shape) == 2:
            d[k] = v[:, :max_length]
    return d


def main(args: WmBaseArgs):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    model = model.to(args.device)
    lm_tokenizer = tokenizer

    class LSTMModel(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            embedded = self.embedding(x)  
            _, (hn, cn) = self.lstm(embedded)
            output = self.fc(hn[-1, :, :])
            return output
        
    seq_length = 30
    vocab_size = 79
    embed_size = 64  
    hidden_size = 128
    output_size = vocab_size
    batch_size = 128

    lstm_model = LSTMModel(vocab_size, embed_size, hidden_size, output_size)

    lstm_model.load_state_dict(torch.load("lstm_model_" + args.language+ ".pth"))
    lstm_model.to(args.device)

    c4_sliced_and_filted = load_from_disk(
        os.path.join(ROOT_PATH, ''))
    
    c4_sliced_and_filted = c4_sliced_and_filted['train'].select(range(args.sample_num))

    lm_message_model = RandomMessageModel(tokenizer=tokenizer,
                                          lm_tokenizer=lm_tokenizer,
                                          delta=args.delta,
                                          message_code_len=args.message_code_len,
                                          device=model.device,
                                          )

    watermark_processor = WmProcessorRandomMessageModel(message_model=lm_message_model,
                                                        tokenizer=tokenizer,
                                                        encode_ratio=args.encode_ratio,
                                                        message=args.message,
                                                        top_k=args.top_k,
                                                        )

    min_length_processor = MinLengthLogitsProcessor(min_length=10000,
                                                    eos_token_id=tokenizer.eos_token_id)
    rep_processor = RepetitionPenaltyLogitsProcessor(penalty=args.repeat_penalty)

    ngram_processor = NoRepeatNGramLogitsProcessor(ngram_size=args.ngram_size) 

    pda_model = PDAMessageModel(tokenizer=tokenizer, pda_model=lstm_model,
                                    delta=args.delta)
    
    pda_processor = PDAProcessorMessageModel(message_model=pda_model,tokenizer=tokenizer, gamma=args.gamma)

    
    logit_processor = LogitsProcessorList(
        [min_length_processor, rep_processor, ngram_processor, watermark_processor, pda_processor])

    results = {'text': [],
               'prefix_and_output_text': [],
               'output_text': [],
               'decoded_message': [],
               'acc': []}


    try:
        for text in tqdm(c4_sliced_and_filted['original_string']):
            tokenized_input = tokenizer(text, return_tensors='pt')
            tokenized_input = truncate(tokenized_input, max_length=args.prompt_length)
            tokenized_input = tokenized_input.to(model.device)

            watermark_processor.start_length = tokenized_input['input_ids'].shape[-1]
            output_tokens = model.generate(**tokenized_input,
                                           temperature=args.temperature,
                                           max_new_tokens=args.generated_length,
                                           num_beams=args.num_beams,
                                           logits_processor=logit_processor)

            output_text = \
                tokenizer.batch_decode(
                    output_tokens[:, tokenized_input["input_ids"].shape[-1]:],
                    skip_special_tokens=True)[0]

            prefix_and_output_text = tokenizer.batch_decode(output_tokens,
                                                            skip_special_tokens=True)[0]

            results['text'].append(text)
            results['output_text'].append(output_text)
            results['prefix_and_output_text'].append(prefix_and_output_text)

            decoded_message = watermark_processor.decode(output_text, disable_tqdm=True)[0]
            available_message_num = args.generated_length // (
                int(args.message_code_len * args.encode_ratio))
            acc = decoded_message[:available_message_num] == args.message[:available_message_num]

            results['decoded_message'].append(decoded_message)
            results['acc'].append(acc)
            print(prefix_and_output_text)
            print(decoded_message,acc)
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        pass

    args_dict = vars(args)
    results['args'] = args_dict
    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=4)