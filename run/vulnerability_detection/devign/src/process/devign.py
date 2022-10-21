import torch.optim as optim
import torch.nn.functional as F

from ..utils import log
from .step import Step
from .model import Net


class Devign(Step):
    def __init__(self,
                 path: str,
                 device: str,
                 model: dict,
                 learning_rate: float,
                 weight_decay: float,
                 loss_lambda: float):
        self.path = path
        self.lr = learning_rate
        self.wd = weight_decay
        self.ll = loss_lambda
        log.log_info('devign', f"LR: {self.lr}; WD: {self.wd}; LL: {self.ll};")
        _model = Net(**model, device=device)
        super().__init__(model=_model,
                         loss_function=lambda o, t: F.binary_cross_entropy(o, t) + F.l1_loss(o, t) * self.ll,
                         optimizer=optim.Adam(_model.parameters(), lr=self.lr, weight_decay=self.wd),
                         )

        self.count_parameters()

    def load(self):
        self.model.load(self.path)

    def save(self):
        self.model.save(self.path)

    def count_parameters(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"The model has {count:,} trainable parameters")
