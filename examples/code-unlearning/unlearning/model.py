import pandas as pd
from tqdm import tqdm
import torch
from torch.nn.functional import softmax, kl_div
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torchmetrics.functional import accuracy
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from dataset import CodeDataset, CodeSecretDataset
from torch.utils.data.distributed import DistributedSampler
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import time


class UnlearningFramework(pl.LightningModule):
    def __init__(self, hparams):
        super(UnlearningFramework, self).__init__()
        self.save_hyperparameters(hparams)

        # Model Initializaion
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path, local_files_only=True)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hparams.model_name_or_path,
            # resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
            attention_dropout=0,
            # pad_token_id=self.tokenizer.pad_token_id,
            local_files_only=True
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = len(self.tokenizer)
        if hasattr(self.model, 'model'):
            print("Model has model.model.embed_tokens.padding_idx")
            self.model.model.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        else:
            print("Model has get_input_embeddings.padding_idx")
            self.model.get_input_embeddings().padding_idx = self.tokenizer.pad_token_id
        if hasattr(self.model, 'lm_head'):
            print("Model has lm_head.padding_idx")
            self.model.lm_head.padding_idx = self.tokenizer.pad_token_id
        self.model.get_output_embeddings().padding_idx = self.tokenizer.pad_token_id
            
        embedding_layer = self.model.get_input_embeddings()
        print(f"padding_idx: {embedding_layer.padding_idx}")

        assert self.model.config.pad_token_id == self.tokenizer.pad_token_id
        assert embedding_layer.num_embeddings == len(self.tokenizer)

        self.validation_step_outputs = []  # storing the outputs of validation steps
        # self.valid_df = None  # Dataframe that stores MA & EL for individual examples
        
        self.best_ma = 1.0
        self.current_ma_mean = 1.0
        self.current_ma_max = 1.0
        
        self.el_n_flag = self.hparams.check_validation_only or False
        self.best_el_n = 1.0
        self.current_el_n_mean = 1.0
        self.current_el_n_max = 1.0
        self.el_n_patience = 5
        self.el_n_counter = 0
        
        # self.best_loss = 0
        # self.current_loss = 0
        
        self.best_epoch = 0

    def forward(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self.model(batch["source_ids"], attention_mask=batch["source_mask"], labels=lm_labels)
        loss, score = outputs[0], outputs[1]
        return loss, score

    def on_train_epoch_start(self):
        self.train_sampler.set_epoch(self.current_epoch)
        # torch.cuda.synchronize()
        # self.start_time = time.time()
        # # Reset memory stats at the start of the epoch
        # torch.cuda.reset_peak_memory_stats(device=self.device)
        # torch.cuda.reset_accumulated_memory_stats(device=self.device)

    def training_step(self, batch, batch_idx):
        loss, score = self(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.eval_secret:
            output = self.validation_secret_forget(batch)
        else:
            output = self.validation_forget(batch)
        self.validation_step_outputs.append(output)
        return output

    def validation_secret_forget(self, batch):
        value_dict = {}
        
        # Loss
        # losses = self.validation_loss(batch)
        # value_dict['loss'] = losses

        # MA
        # mas = self.validation_ma(batch)
        # value_dict['ma'] = mas
        mas = self.validation_secret_ma(batch)
        value_dict['ma_mean'] = mas.mean()
        value_dict['ma_max'] = mas.max()

        # EL
        if self.el_n_flag:
            # els = self.validation_el(batch)
            # value_dict[f'el_{self.hparams.el_n}-gram'] = els
            els = self.validation_secret_el(batch)
            value_dict[f'el_{self.hparams.el_n}-gram_mean'] = els.mean()
            value_dict[f'el_{self.hparams.el_n}-gram_max'] = els.max()
        
        # value_dict['doc_id'] = batch['doc_id']
        return value_dict
    
    def validation_forget(self, batch):
        value_dict = {}
        
        # Loss
        # losses = self.validation_loss(batch)
        # value_dict['loss'] = losses

        # MA
        mas = self.validation_ma(batch)
        value_dict['ma_mean'] = mas.mean()
        value_dict['ma_max'] = mas.max()

        # EL
        if self.el_n_flag:
            els = self.validation_el(batch)
            value_dict[f'el_{self.hparams.el_n}-gram_mean'] = els.mean()
            value_dict[f'el_{self.hparams.el_n}-gram_max'] = els.max()
        
        # value_dict['doc_id'] = batch['doc_id']
        return value_dict
    
    def validation_loss(self, batch):
        _, score = self(batch)
        # Recalculate loss individually
        shift_logits = score[..., :-1, :].contiguous().squeeze()
        shift_labels = batch['target_ids'][..., 1:].contiguous().squeeze()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_no_reduce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # Reduce along sequence only, leave batch
        if len(batch['target_ids'].shape) > 1:
            loss_no_reduce = loss_no_reduce.view(batch['target_ids'].shape[0], -1)  # (batch, seq_len)
        else:
            loss_no_reduce = torch.unsqueeze(loss_no_reduce, 0)
        individual_losses = []
        for seq_loss in loss_no_reduce:
            individual_loss = seq_loss[seq_loss != 0].mean()
            individual_losses.append(individual_loss)
        individual_losses = torch.stack(individual_losses)
        # individual_perplexities = torch.exp(individual_losses)
        return individual_losses
    
    def validation_secret_ma(self, batch):
        doc_ids = batch['doc_id']

        secret_token_ids = torch.cat([self.valid_dataset_secret_token_ids[doc_id.item()] for doc_id in doc_ids], dim=0)
        secret_prefix_token_ids = torch.cat([self.valid_dataset_secret_prefix_token_ids[doc_id.item()] for doc_id in doc_ids], dim=0)

        preds = []
        for i in range(secret_token_ids.shape[-1]):
            prompt = torch.cat([secret_prefix_token_ids, secret_token_ids[..., :i]], dim=-1)
            # pred = self.model.generate(prompt, max_length=secret_prefix_token_ids.shape[-1] + i + 1)[..., -1]
            pred = self.model.generate(prompt, max_new_tokens=1, pad_token_id=self.tokenizer.pad_token_id)[..., -1]
            preds.append(torch.squeeze(pred))
            # del prompt, pred
            # torch.cuda.empty_cache()

        preds = torch.t(torch.stack(preds))
        secret_token_ids[secret_token_ids == self.tokenizer.pad_token_id] = -100
        secret_mas = accuracy(preds, secret_token_ids, task='multiclass', num_classes=len(self.tokenizer.get_vocab()), ignore_index=-100, multidim_average='samplewise')
        # secret_ma_global = accuracy(preds, secret_token_ids, task='multiclass', num_classes=len(self.tokenizer.get_vocab()), ignore_index=-100, multidim_average='global')

        return secret_mas

    def validation_ma(self, batch):
        input_ids = batch['source_ids']
        max_len = self.hparams.target_length

        labels, preds = [], []
        # for i in tqdm(range(1, max_len), desc="Calculating MA", position=0):
        for i in range(1, max_len):
            label = input_ids[..., i]
            prompt = input_ids[..., :i]
            # pred = self.model.generate(prompt, max_length=i + 1)[:, -1]
            pred = torch.argmax(self.model(prompt).logits[:, -1, :], dim=-1)
            labels.append(torch.squeeze(label))
            preds.append(torch.squeeze(pred))

        preds = torch.t(torch.stack(preds))
        labels = torch.t(torch.stack(labels))
        
        accs = []
        if len(preds.shape) == 1:
            preds = torch.unsqueeze(preds, 0)
            labels = torch.unsqueeze(labels, 0)
        for pred, label in zip(preds, labels):
            acc = accuracy(pred, label, task='multiclass', num_classes=len(self.tokenizer.get_vocab()), ignore_index=-100)
            accs.append(acc)
        accs = torch.stack(accs)

        return accs
    
    def validation_secret_el(self, batch):
        doc_ids = batch['doc_id']

        secret_token_ids = torch.cat([self.valid_dataset_secret_token_ids[doc_id.item()] for doc_id in doc_ids], dim=0)
        secret_prefix_token_ids = torch.cat([self.valid_dataset_secret_prefix_token_ids[doc_id.item()] for doc_id in doc_ids], dim=0)
        
        secret_num = secret_token_ids.shape[0]
        secret_max_len = secret_token_ids.shape[-1]
        secret_prefix_max_len = secret_prefix_token_ids.shape[-1]
        secret_valid_token_num = torch.sum(secret_token_ids != self.tokenizer.pad_token_id, dim=-1)

        numerator = torch.zeros(secret_num).to(self.device)
        # for i in tqdm(list(reversed(range(secret_max_len - (self.hparams.el_n - 1))))):
        for i in list(reversed(range(secret_max_len - (self.hparams.el_n - 1)))):
            label = secret_token_ids[..., i:secret_max_len]
            if torch.sum(torch.sum(label != self.tokenizer.pad_token_id, dim=-1) >= self.hparams.el_n) == 0:
                # del label
                # torch.cuda.empty_cache()
                continue
            prompt = torch.cat([secret_prefix_token_ids, secret_token_ids[..., :i]], dim=-1)
            # pred = self.model.generate(prompt, max_length=secret_prefix_max_len + secret_max_len)[..., secret_prefix_max_len + i:]
            pred = self.model.generate(prompt, max_new_tokens=secret_max_len - i, pad_token_id=self.tokenizer.pad_token_id)[..., secret_prefix_max_len + i:]
            pred_ngram = self.ngram_of_tensor(pred, self.hparams.el_n)
            label_ngram = self.ngram_of_tensor(label, self.hparams.el_n)
            valid_ngram_flag = torch.sum(label_ngram == self.tokenizer.pad_token_id, dim=-1) == 0
            valid_ngram_num = torch.sum(valid_ngram_flag, dim=-1)
            pred_duplication_index = torch.arange(pred_ngram.shape[-2]).unsqueeze(1).repeat(1, pred_ngram.shape[-2])
            '''
            tensor([[0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3]])
            '''
            label_duplication_index = torch.arange(pred_ngram.shape[-2]).unsqueeze(0).repeat(pred_ngram.shape[-2], 1)
            '''
            tensor([[0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3]])
            '''
            ngram_match = torch.sum(pred_ngram[:, pred_duplication_index, :] != label_ngram[:, label_duplication_index, :], dim=-1) == 0
            ngram_has_no_pad_token_id = torch.sum(label_ngram[:, label_duplication_index, :] == self.tokenizer.pad_token_id, dim=-1) == 0
            ngram_has_no_pad_token_id = torch.logical_and(ngram_has_no_pad_token_id, torch.transpose(ngram_has_no_pad_token_id, -2, -1))
            valid_ngram_match = torch.sum(torch.logical_and(ngram_match, ngram_has_no_pad_token_id), dim=-1) > 0
            valid_ngram_match_num = torch.sum(valid_ngram_match, dim=-1)
            # print(torch.sum(valid_ngram_match_num[valid_ngram_num == 0] != 0))
            valid_ngram_match_num[valid_ngram_num == 0] = 0  # prevent NAN values
            valid_ngram_num[valid_ngram_num == 0] = 1  # prevent NAN values
            p_acc = valid_ngram_match_num.float() / valid_ngram_num
            numerator += p_acc
            # del prompt, pred, label, pred_ngram, label_ngram, p_tp, p_accFalse
            # torch.cuda.empty_cache()
        temp = secret_valid_token_num - (self.hparams.el_n - 1)
        numerator[secret_valid_token_num < self.hparams.el_n] = 0  # prevent NAN values
        temp[secret_valid_token_num < self.hparams.el_n] = 1  # prevent NAN values
        el_score = numerator / temp
        
        # special cases: token num < n
        if torch.sum(secret_valid_token_num < self.hparams.el_n) > 0:
            special_label = secret_token_ids[secret_valid_token_num < self.hparams.el_n, ...]
            special_prompt = secret_prefix_token_ids[secret_valid_token_num < self.hparams.el_n, ...]
            if len(special_label.shape) == 1:
                special_label = torch.unsqueeze(special_label, 0)
                special_prompt = torch.unsqueeze(special_prompt, 0)
            # special_pred = self.model.generate(special_prompt, max_length=secret_prefix_max_len + secret_max_len)[..., secret_prefix_max_len:]
            special_pred = self.model.generate(special_prompt, max_new_tokens=secret_max_len, pad_token_id=self.tokenizer.pad_token_id)[..., secret_prefix_max_len:]
            special_label[special_label == self.tokenizer.pad_token_id] = -100
            special_mas = accuracy(special_pred, special_label, task='multiclass', num_classes=len(self.tokenizer.get_vocab()), ignore_index=-100, multidim_average='samplewise')
            el_score[secret_valid_token_num < self.hparams.el_n] += (special_mas == 1).float()
        
        return el_score

    def validation_el(self, batch):
        input_ids = batch['source_ids']
        max_len = self.hparams.target_length

        batch_size = input_ids.shape[0]
        
        '''
        Jang et al.'s original implementation of calculating EL is slow for big batch size. 
        To address this, we have switched from loop operations to matrix operations to 
        reduce the computational time. However, this adjustment may lead to increased memory
        consumption. We recommend selecting the implementation that best suits your needs. 
        Please refer to the commented codes for access to the original implementation.
        '''
        # numerator = [0] * batch_size
        # for i in tqdm(list(reversed(range(1, max_len - (self.hparams.el_n - 1)))), desc="Calculating EL", position=0):
        #     label = input_ids[..., i:max_len]
        #     prompt = input_ids[..., :i]
        #     # pred = self.model.generate(prompt, max_length=max_len)[..., i:]
        #     temp_inputs = prompt
        #     for _ in range(i, max_len):
        #         next_tokens = torch.argmax(self.model(temp_inputs).logits[:, -1, :], dim=-1)
        #         temp_inputs = torch.cat([temp_inputs, next_tokens.unsqueeze(-1)], dim=-1)
        #     pred = temp_inputs[..., i:]
        #     for example_idx in range(batch_size):
        #         p, l = pred[example_idx], label[example_idx]
        #         # extraction likelihood
        #         p_ngram = self.ngram_of_1D_tensor(p, self.hparams.el_n)
        #         l_ngram = self.ngram_of_1D_tensor(l, self.hparams.el_n)
        #         l_unique = set(l_ngram)
        #         p_tp = [i for i in p_ngram if i in l_unique]
        #         try:
        #             p_acc = len(p_tp) / len(l_ngram)
        #             numerator[example_idx] += p_acc
        #         except ZeroDivisionError:  # n-gram isn't defined
        #             pass
        # el_score = [0] * batch_size
        # for i, _ in enumerate(numerator):
        #     el_score[i] = numerator[i] / (max_len - 1 - (self.hparams.el_n - 1))
        # el_score = torch.Tensor(el_score).to(self.device)
        numerator = torch.zeros(batch_size).to(self.device)
        # for i in tqdm(list(reversed(range(1, max_len - (self.hparams.el_n - 1)))), desc="Calculating EL", position=0):
        for i in list(reversed(range(1, max_len - (self.hparams.el_n - 1)))):
            label = input_ids[..., i:max_len]
            prompt = input_ids[..., :i]
            # pred = self.model.generate(prompt, max_length=max_len)[..., i:]
            temp_inputs = prompt
            for _ in range(i, max_len):
                next_tokens = torch.argmax(self.model(temp_inputs).logits[:, -1, :], dim=-1)
                temp_inputs = torch.cat([temp_inputs, next_tokens.unsqueeze(-1)], dim=-1)
            pred = temp_inputs[..., i:]
            pred_ngram = self.ngram_of_tensor(pred, self.hparams.el_n)
            label_ngram = self.ngram_of_tensor(label, self.hparams.el_n)
            pred_duplication_index = torch.arange(pred_ngram.shape[-2]).unsqueeze(1).repeat(1, pred_ngram.shape[-2])
            label_duplication_index = torch.arange(pred_ngram.shape[-2]).unsqueeze(0).repeat(pred_ngram.shape[-2], 1)
            p_tp = torch.sum(pred_ngram[:, pred_duplication_index, :] != label_ngram[:, label_duplication_index, :], dim=-1) == 0
            p_tp = torch.sum(p_tp, dim=-1) > 0
            p_acc = torch.sum(p_tp, dim=-1).float() / p_tp.shape[-1]
            numerator += p_acc
        el_score = numerator / (max_len - 1 - (self.hparams.el_n - 1))

        return el_score

    # Reduce results from gpus to a single dataframe + determine early stopping
    def on_validation_epoch_end(self):
        # assert False
        log_col_name = f'{(self.current_epoch + 1):02d}'

        outputs = self.validation_step_outputs
        gathered_outputs = self.all_gather(outputs)  # Reduce results from gpus
        keys = gathered_outputs[0].keys()
        full_output = {k: [] for k in keys}

        # gather all outputs
        for out in gathered_outputs:
            for k in keys:
                full_output[k].append(torch.flatten(out[k]))

        # refactor into pandas favorable format
        for k in keys:
            full_output[k] = torch.cat(full_output[k], dim=-1).detach().cpu().numpy()
        
        df = pd.DataFrame(full_output)
        # df = df.set_index('doc_id')
        new_column_name = []
        for column_name in df.columns:
            new_column_name.append(f'{column_name}_{log_col_name}')
        df.columns = new_column_name
        # self.valid_df = self.valid_df.combine_first(df)
        
        self.current_ma_mean = df[f'ma_mean_{log_col_name}'].mean()  # max!!!
        if self.el_n_flag:
            self.current_el_n_mean = df[f'el_{self.hparams.el_n}-gram_mean_{log_col_name}'].mean()  # max!!!
        # self.current_loss = df[f'loss_{log_col_name}'].mean()
        
        if self.local_rank == 0:
            # print(f"Current: acc={self.current_acc} el_{self.hparams.el_n}-gram={self.current_el_n} loss={self.current_loss} epoch={self.current_epoch + 1}")
            print(f"Current: ma={self.current_ma_mean} el_{self.hparams.el_n}-gram={self.current_el_n_mean} epoch={self.current_epoch + 1}")

        self.validation_step_outputs.clear()

    # def on_validation_end(self):
    #     if self.local_rank == 0:
    #         if self.hparams.valid_save_path != '':
    #             self.valid_df.to_csv(self.hparams.valid_save_path)

    def save_model(self):
        pass
    
    def on_train_epoch_end(self):
        # torch.cuda.synchronize()
        # self.end_time = time.time()
        # total_time = self.end_time - self.start_time
        # print(f"Epoch run time: {total_time:.5f} seconds")
        # # Log memory stats at the end of the epoch
        # allocated_memory = torch.cuda.memory_allocated(device=self.device) / 1024**3  # convert to MB
        # max_memory = torch.cuda.max_memory_allocated(device=self.device) / 1024**3  # convert to MB
        # print(f"Device: {self.device}", f"Allocated memory at end of epoch: {allocated_memory} MB", f"Peak memory during this epoch: {max_memory} MB")

        if self.current_ma_mean <= self.best_ma:  # max!!!
            if self.el_n_flag:
                if self.current_el_n_mean <= self.best_el_n:
                    self.best_el_n = self.current_el_n_mean
                    self.el_n_counter = 0
                    self.best_ma = self.current_ma_mean
                    # self.best_loss = self.current_loss
                    self.best_epoch = self.current_epoch + 1
                    self.save_model()
                else:
                    self.el_n_counter += 1
            else:
                self.best_ma = self.current_ma_mean
                # self.best_loss = self.current_loss
                self.best_epoch = self.current_epoch + 1
                self.save_model()
                
        if self.local_rank == 0:
            # print(f"Best: ma={self.best_ma} el_{self.hparams.el_n}-gram={self.best_el_n} loss={self.best_loss} epoch={self.best_epoch}")
            print(f"Best: ma={self.best_ma} el_{self.hparams.el_n}-gram={self.best_el_n} epoch={self.best_epoch}")
        
        # check early stopping criteria
        if self.current_ma_mean < self.hparams.ma_threshold:
            if self.hparams.el_n != 10:
                self.trainer.should_stop = True
            else:
                if self.el_n_flag:
                    if self.current_el_n_mean < self.hparams.el10_threshold or self.el_n_counter >= self.el_n_patience:
                        self.trainer.should_stop = True
        
        if self.current_ma_mean <= (self.hparams.ma_threshold + self.hparams.margin):
            if self.hparams.el_n == 10 and not self.el_n_flag:
                self.el_n_flag = True

    def configure_optimizers(self):
        parameters = self.model.parameters()
        if self.hparams.strategy in ['deepspeed_stage_2']:
            optimizer = deepspeed.ops.adam.FusedAdam(parameters, lr=self.hparams.learning_rate, betas=(0.9, 0.98))
        elif self.hparams.strategy in ['deepspeed_stage_2_offload']:
            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(parameters, lr=self.hparams.learning_rate, betas=(0.9, 0.98))
        else:
            raise Exception("Please select correct optimization strategy.")
        return [optimizer]

    def train_dataloader(self):
        # train_dataset = CodeDataset(
        #     tokenizer=self.tokenize, dataset_name=self.hparams.train_set,
        #     type_path="train", input_length=self.hparams.target_length,
        #     output_length=self.hparams.target_length, args=self.hparams
        # )
        train_dataset = CodeSecretDataset(
            tokenizer=self.tokenizer, dataset_name=self.hparams.train_set,
            type_path="train", args=self.hparams
        )
        self.train_sampler = DistributedSampler(
            train_dataset, num_replicas=self.hparams.ngpu,
            rank=torch.distributed.get_rank(),
            shuffle=True, seed=self.hparams.seed
        )
        dataloader = DataLoader(
            train_dataset, sampler=self.train_sampler,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers
        )
        return dataloader

    def val_dataloader(self):
        if self.hparams.eval_secret:
            valid_dataset = CodeSecretDataset(
                tokenizer=self.tokenizer, dataset_name=self.hparams.valid_set,
                type_path="valid", args=self.hparams
            )
            self.valid_dataset_secret_token_ids = {}
            self.valid_dataset_secret_prefix_token_ids = {}
            for data_idx in range(len(valid_dataset)):
                data = valid_dataset.dataset[data_idx]
                self.valid_dataset_secret_token_ids[data['doc_id']] = torch.LongTensor(data['secret_token_ids']).to(self.device)
                self.valid_dataset_secret_prefix_token_ids[data['doc_id']] = torch.LongTensor(data['secret_prefix_token_ids']).to(self.device)
        else:
            valid_dataset = CodeDataset(
                tokenizer=self.tokenizer, dataset_name=self.hparams.valid_set,
                type_path="valid", input_length=self.hparams.target_length,
                output_length=self.hparams.target_length, args=self.hparams
            )
        
        # if self.valid_df is None:
            # self.valid_df = valid_dataset.dataset.set_index('doc_id')
        dataloader = DataLoader(
            valid_dataset, batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers, shuffle=False
        )
        return dataloader

    def ngram_of_tensor(self, X, n):
        return torch.stack([X[..., i:i + n] for i in range(X.shape[-1] - n + 1)], dim=-2)

    def ngram_of_1D_tensor(self, X, n):
        grams = [tuple(X[i:i + n].tolist()) for i in range(X.shape[0] - n + 1)]
        return grams


class GradientAscent(UnlearningFramework):
    def __init__(self, hparams):
        super(GradientAscent, self).__init__(hparams)
        
    def training_step(self, batch, batch_idx):
        loss, score = self(batch)
        loss = -1 * loss
        return loss
    
    def save_model(self):
        if self.local_rank == 0:
            model_name = self.hparams.model_name_or_path.split('/')[-1]
            forgot_data_name = self.hparams.train_set.split('/')[-1]
            save_path = f'ckpts/{model_name}/{forgot_data_name}_GA'
            save_path = save_path + f'_lr{self.hparams.learning_rate}'
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)


class ControllableUnlearning(UnlearningFramework):
    def __init__(self, hparams):
        super(ControllableUnlearning, self).__init__(hparams)

        if 'codeparrot' in self.hparams.model_name_or_path:
            control_model_name_or_path = 'codeparrot/codeparrot'
        elif 'codegen-350M' in self.hparams.model_name_or_path:
            control_model_name_or_path = 'Salesforce/codegen-350M-mono'
        elif 'codegen-2B' in self.hparams.model_name_or_path:
            control_model_name_or_path = 'Salesforce/codegen-2B-mono'
        elif 'codegen-6B' in self.hparams.model_name_or_path:
            control_model_name_or_path = 'Salesforce/codegen-6B-mono'
        elif 'codegen-16B' in self.hparams.model_name_or_path:
            control_model_name_or_path = 'Salesforce/codegen-16B-mono'
        elif 'Qwen' in self.hparams.model_name_or_path:
            control_model_name_or_path = 'Qwen/Qwen2.5-Coder-7B'

        if self.hparams.unlearning_mode == "PECU":
            self.control_model = self.model
        else:
            self.control_model = AutoModelForCausalLM.from_pretrained(
                control_model_name_or_path,
                # resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                attention_dropout=0,
                # pad_token_id=self.tokenizer.pad_token_id,
                local_files_only=True
            )
            self.control_model.resize_token_embeddings(len(self.tokenizer))
            
            self.control_model.config.pad_token_id = self.tokenizer.pad_token_id
            self.control_model.config.vocab_size = len(self.tokenizer)
            if hasattr(self.control_model, 'model'):
                print("Model has model.model.embed_tokens.padding_idx")
                self.control_model.model.embed_tokens.padding_idx = self.tokenizer.pad_token_id
            else:
                print("Model has get_input_embeddings.padding_idx")
                self.control_model.get_input_embeddings().padding_idx = self.tokenizer.pad_token_id
            if hasattr(self.control_model, 'lm_head'):
                print("Model has lm_head.padding_idx")
                self.control_model.lm_head.padding_idx = self.tokenizer.pad_token_id

        for param in self.control_model.parameters():
            param.requires_grad = False
                
    def on_train_epoch_start(self):
        self.train_sampler.set_epoch(self.current_epoch)
        self.control_sampler.set_epoch(self.current_epoch)
        # torch.cuda.synchronize()
        # self.start_time = time.time()
        # # Reset memory stats at the start of the epoch
        # torch.cuda.reset_peak_memory_stats(device=self.device)
        # torch.cuda.reset_accumulated_memory_stats(device=self.device)
                
    def training_step(self, batch, batch_idx):
        data_idx = batch_idx // self.hparams.gradient_accumulation_steps
        
        # print(data_idx, batch_idx, batch['doc_id'])
        input_ids, attention_mask = batch["source_ids"], batch["source_mask"]
        scores = self.model(input_ids, attention_mask=attention_mask).logits
        shift_logits = scores[..., :-1, :].contiguous().squeeze().float()
        shift_logits_normalized = shift_logits - torch.max(shift_logits, dim=-1, keepdim=True).values  # Prevents numerical problems caused by excessive input values
        probs = softmax(shift_logits_normalized, dim=-1)
        with torch.no_grad():
            ref_scores = self.control_model(input_ids, attention_mask=attention_mask).logits
            ref_shift_logits = ref_scores[..., :-1, :].contiguous().squeeze().float()
            ref_shift_logits_normalized = ref_shift_logits - torch.max(ref_shift_logits, dim=-1, keepdim=True).values  # Prevents numerical problems caused by excessive input values
            ref_probs = softmax(ref_shift_logits_normalized, dim=-1)
        kl_loss = kl_div((probs + 1e-10).log(), ref_probs + 1e-10, reduction='batchmean')
        
        if data_idx == 0:
            loss, score = self(batch)
            loss = -1 * loss + self.hparams.control_lambda * (-1) * kl_loss
            # print("Forgot Loss:", loss)
        else:
            loss = self.hparams.control_lambda * self.hparams.control_alpha * kl_loss
            # print("Control Loss:", loss)
        
        return loss
    
    def save_model(self):
        if self.local_rank == 0:
            model_name = self.hparams.model_name_or_path.split('/')[-1]
            forgot_data_name = self.hparams.train_set.split('/')[-1]
            save_path = f'ckpts/{model_name}/{forgot_data_name}_CU'
            save_path = save_path + f'_lr{self.hparams.learning_rate}_alpha{self.hparams.control_alpha}_lambda{self.hparams.control_lambda}'
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
    
    def train_dataloader(self):
        dataloaders = []
        # train_dataset = CodeDataset(
        #     tokenizer=self.tokenizer, dataset_name=self.hparams.train_set,
        #     type_path="train", input_length=self.hparams.target_length,
        #     output_length=self.hparams.target_length, args=self.hparams
        # )
        train_dataset = CodeSecretDataset(
            tokenizer=self.tokenizer, dataset_name=self.hparams.train_set,
            type_path="train", args=self.hparams
        )
        self.train_sampler = DistributedSampler(
            train_dataset, num_replicas=self.hparams.ngpu,
            rank=torch.distributed.get_rank(),
            shuffle=True, seed=self.hparams.seed
        )
        dataloaders.append(DataLoader(
            train_dataset, sampler=self.train_sampler,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers
        ))
        
        control_dataset = CodeDataset(
            tokenizer=self.tokenizer, dataset_name=self.hparams.control_set,
            type_path="control", input_length=self.hparams.target_length,
            output_length=self.hparams.target_length, args=self.hparams
        )
        self.control_sampler = DistributedSampler(
            control_dataset, num_replicas=self.hparams.ngpu,
            rank=torch.distributed.get_rank(),
            shuffle=True, seed=self.hparams.seed
        )
        dataloaders.append(DataLoader(
            control_dataset, sampler=self.control_sampler,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers
        ))
        
        combined_loader = CombinedLoader(dataloaders, 'sequential')
        _ = iter(combined_loader)
        
        # for batch, batch_idx, dataloader_idx in combined_loader:
        #     print(f"{batch['doc_id']}, {batch_idx=}, {dataloader_idx=}")
        
        return combined_loader


class SensitivitySelectiveGA(GradientAscent):
    def __init__(self, hparams):
        super(SensitivitySelectiveGA, self).__init__(hparams)
        
    def training_step(self, batch, batch_idx):
        _, score = self(batch)
        
        shift_logits = score[..., :-1, :].contiguous()
        shift_labels = batch['target_ids'][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_no_reduce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_no_reduce = loss_no_reduce.view(shift_labels.size())
        
        shift_secret_mask = batch['secret_mask'][..., 1:].contiguous()
        secret_loss = loss_no_reduce[shift_secret_mask].mean()
        normal_loss = loss_no_reduce[~shift_secret_mask].mean()

        loss = -1 * secret_loss + self.hparams.select_gamma * normal_loss
        return loss
    
    def save_model(self):
        if self.local_rank == 0:
            model_name = self.hparams.model_name_or_path.split('/')[-1]
            forgot_data_name = self.hparams.train_set.split('/')[-1]
            save_path = f'ckpts/{model_name}/{forgot_data_name}_SSGA'
            save_path = save_path + f'_lr{self.hparams.learning_rate}_gamma{self.hparams.select_gamma}'
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)


class SensitivitySelectiveCU(ControllableUnlearning):
    def __init__(self, hparams):
        super(SensitivitySelectiveCU, self).__init__(hparams)

    def training_step(self, batch, batch_idx):
        data_idx = batch_idx // self.hparams.gradient_accumulation_steps
        
        if data_idx == 0:
            shift_secret_mask = batch['secret_mask'][..., 1:].contiguous()
        
        # print(data_idx, batch_idx, batch['doc_id'])
        input_ids, attention_mask = batch["source_ids"], batch["source_mask"]
        scores = self.model(input_ids, attention_mask=attention_mask).logits
        shift_logits = scores[..., :-1, :].contiguous().float()
        if data_idx == 0:
            shift_logits = shift_logits[shift_secret_mask]
        shift_logits_normalized = shift_logits - torch.max(shift_logits, dim=-1, keepdim=True).values  # Prevents numerical problems caused by excessive input values
        probs = softmax(shift_logits_normalized, dim=-1)
        with torch.no_grad():
            ref_scores = self.control_model(input_ids, attention_mask=attention_mask).logits
            ref_shift_logits = ref_scores[..., :-1, :].contiguous().float()
            if data_idx == 0:
                ref_shift_logits = ref_shift_logits[shift_secret_mask]
            ref_shift_logits_normalized = ref_shift_logits - torch.max(ref_shift_logits, dim=-1, keepdim=True).values  # Prevents numerical problems caused by excessive input values
            ref_probs = softmax(ref_shift_logits_normalized, dim=-1)
        kl_loss = kl_div((probs + 1e-10).log(), ref_probs + 1e-10, reduction='batchmean')
        
        if data_idx == 0:
            _, score = self(batch)
            
            shift_logits = score[..., :-1, :].contiguous()
            shift_labels = batch['target_ids'][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss_no_reduce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_no_reduce = loss_no_reduce.view(shift_labels.size())
            
            secret_loss = loss_no_reduce[shift_secret_mask].mean()
            normal_loss = loss_no_reduce[~shift_secret_mask].mean()

            loss = -1 * secret_loss + self.hparams.select_gamma * normal_loss
            loss = loss + self.hparams.control_lambda * (-1) * kl_loss
            # print("Forgot Loss:", loss)
        else:
            loss = self.hparams.control_lambda * self.hparams.control_alpha * kl_loss
            # print("Control Loss:", loss)
        
        return loss
    
    def save_model(self):
        if self.local_rank == 0:
            model_name = self.hparams.model_name_or_path.split('/')[-1]
            forgot_data_name = self.hparams.train_set.split('/')[-1]
            save_path = f'ckpts/{model_name}/{forgot_data_name}_SSCU'
            save_path = save_path + f'_lr{self.hparams.learning_rate}_alpha{self.hparams.control_alpha}_lambda{self.hparams.control_lambda}_gamma{self.hparams.select_gamma}'
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)


class PEFTControllableUnlearning(ControllableUnlearning):
    def __init__(self, hparams):
        super(PEFTControllableUnlearning, self).__init__(hparams)
        
        if self.hparams.check_validation_only:
            self.model = PeftModel.from_pretrained(
                self.model, self.hparams.peft_model_name_or_path,
                local_files_only=True
            )
            self.model = self.model.merge_and_unload()
        else:
            print(self.model)
            if 'codeparrot' in self.hparams.model_name_or_path:
                target_modules=["c_attn", "c_proj", "c_fc"]
            elif 'codegen' in self.hparams.model_name_or_path:
                target_modules=["qkv_proj", "out_proj", "fc_in", "fc_out"]
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, target_modules=target_modules, inference_mode=False, 
                r=self.hparams.lora_r, lora_alpha=self.hparams.lora_alpha, lora_dropout=self.hparams.lora_dropout
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
    def save_model(self):
        if self.local_rank == 0:
            model_name = self.hparams.model_name_or_path.split('/')[-1]
            forgot_data_name = self.hparams.train_set.split('/')[-1]
            save_path = f'ckpts/{model_name}/{forgot_data_name}_PECU'
            save_path = save_path + f'_lr{self.hparams.learning_rate}_alpha{self.hparams.control_alpha}_lambda{self.hparams.control_lambda}'
            save_path = save_path + f'_r{self.hparams.lora_r}_lalpha{self.hparams.lora_alpha}_ldp{self.hparams.lora_dropout}'
            self.model.save_pretrained(save_path)
