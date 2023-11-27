import dataclasses
from dataclasses import dataclass
from typing import List


class Stat:
    def __init__(self, outs=None, loss=0.0, acc=0.0, labels=None):
        if labels is None:
            labels = []
        if outs is None:
            outs = []
        self.outs = outs
        self.labels = labels
        self.loss = loss
        self.acc = acc

    def __add__(self, other):
        return Stat(self.outs + other.outs, self.loss + other.loss, self.acc + other.acc, self.labels + other.labels)

    def __str__(self):
        return f"Loss: {round(self.loss, 4)}; Acc: {round(self.acc, 4)};"


@dataclass
class Stats:
    name: str
    results: List[Stat] = dataclasses.field(default_factory=list)
    total: Stat = Stat()

    def __call__(self, stat):
        self.total += stat
        self.results.append(stat)

    def __str__(self):
        return f"{self.name} {self.mean()}"

    def __len__(self):
        return len(self.results)

    def mean(self):
        res = Stat()
        res += self.total
        res.loss /= len(self)
        res.acc /= len(self)

        return res

    def loss(self):
        return self.mean().loss

    def acc(self):
        return self.mean().acc

    def outs(self):
        return self.total.outs

    def labels(self):
        return self.total.labels
