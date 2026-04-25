from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import itertools
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


class FineTuneModel(nn.Module):
    def __init__(self, base_model):
        super(FineTuneModel, self).__init__()
        self.base_model = base_model
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        hidden_size = self.base_model.config.hidden_size
        
        self.fc = nn.Linear(hidden_size * 2, 1)
        nn.init.xavier_uniform_(self.fc.weight) 
    def forward(self, current_input_ids, current_attention_mask,
                       code_input_ids, code_attention_mask):

        current_outputs = self.base_model(input_ids=current_input_ids,
                                          attention_mask=current_attention_mask,
                                          output_hidden_states=True,
                                          use_cache=False)
        code_outputs = self.base_model(input_ids=code_input_ids,
                                       attention_mask=code_attention_mask,
                                       output_hidden_states=True,
                                       use_cache=False)

        current_last_token = current_outputs.hidden_states[-1][:, -1, :]
        code_last_token = code_outputs.hidden_states[-1][:, -1, :]

        combined = torch.cat([current_last_token, code_last_token], dim=-1)
        

        value = self.fc(combined.float())
        return value



class TDDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        current_sentence, next_sentence, reward, code = self.data_list[idx]
        current_enc = self.tokenizer(
            current_sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        next_enc = self.tokenizer(
            next_sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        code_enc = self.tokenizer(
            code,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "current_input_ids": current_enc["input_ids"].squeeze(0),
            "current_attention_mask": current_enc["attention_mask"].squeeze(0),
            "next_input_ids": next_enc["input_ids"].squeeze(0),
            "next_attention_mask": next_enc["attention_mask"].squeeze(0),
            "code_input_ids": code_enc["input_ids"].squeeze(0),
            "code_attention_mask": code_enc["attention_mask"].squeeze(0),
            "reward": torch.tensor(reward, dtype=torch.float)
        }

BEGIN_LINE = 0
END_LINE = None
BATCH_SIZE = 256

if __name__ == "__main__":
    dist.init_process_group(backend="nccl") 
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{rank}")

    model_name = "Base-Model-Path" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'left'
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    base_model.to(device)
    base_model.config.use_cache = False

    model = FineTuneModel(base_model).to(device)
    model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)


    gamma = 0.9
    learning_rate = 5e-6
    num_epochs = 3  

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    data_list = []
    import json
    
    
    file_name = "Training-Data"
    with open(file_name,"r") as f:
        for line in itertools.islice(f, BEGIN_LINE, END_LINE):
            p = json.loads(line)
            current_response = ""
            for i in range(len(p["sentence"]) - 1):
                current_response += p["sentence"][i] + "."
                data_list.append((current_response,current_response + p["sentence"][i + 1] + ".",p["value"][i],p["code"]))


    dataset = TDDataset(data_list, tokenizer, max_length=512)
    print("---------length of dataset---------------")
    print(len(dataset))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE // world_size, sampler=sampler) 
    print("\n\n--------------length of dataloder----------------")
    print(len(dataloader))

    model.train()
    it_loss = []
    for epoch in range(num_epochs):
        for idx,batch in tqdm(enumerate(dataloader)):
            current_input_ids = batch["current_input_ids"].to(device)
            current_attention_mask = batch["current_attention_mask"].to(device)
            next_input_ids = batch["next_input_ids"].to(device)
            next_attention_mask = batch["next_attention_mask"].to(device)

            code_input_ids, code_attention_mask = batch["code_input_ids"].to(device), batch["code_attention_mask"].to(device)

            reward = batch["reward"].to(device)


            V_current = model(current_input_ids, current_attention_mask,code_input_ids, code_attention_mask)
            V_next = model(next_input_ids, next_attention_mask,code_input_ids,code_attention_mask)

            td_target = reward.unsqueeze(1) + gamma * V_next.detach()

            loss = criterion(V_current, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"{idx}/{len(dataloader)}",flush=True)
            if idx % 5 == 4:
                if rank == 0:
                    it_loss.append(loss.item())
                    

        print(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.4f}",flush = True)
    if rank == 0:
        torch.save(model.module.state_dict(), "HALLU_Value.pt")

    if rank == 0:
        OUTPUT_FILE = "loss.log"
        with open(OUTPUT_FILE,"w") as output_file:
            output_file.write(str(it_loss))
 
