import torch
import torch.nn as nn
import torch.nn.functional as F

class AuditModel(nn.Module):
    def __init__(self, input_size, hidden_states):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_states),
            nn.Tanh(),
            nn.Linear(hidden_states, 2)
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, input, labels=None):
        logits = self.model(input)
        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits, loss
        else:
            return logits
        
    def predict(self, input):
        pred = F.softmax(self.forward(input), dim=-1)
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return ans, pred.cpu()