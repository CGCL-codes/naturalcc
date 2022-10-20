from ..utils.objects.metrics import Metrics
import torch
import time
from ..utils import log as logger


class Train(object):
    def __init__(self, step, epochs, verbose=True):
        self.epochs = epochs
        self.step = step
        self.history = History()
        self.verbose = verbose

    def __call__(self, train_loader_step, val_loader_step=None, early_stopping=None):
        for epoch in range(self.epochs):
            self.step.train()
            train_stats = train_loader_step(self.step)
            self.history(train_stats, epoch + 1)

            if val_loader_step is not None:
                with torch.no_grad():
                    self.step.eval()
                    val_stats = val_loader_step(self.step)
                    self.history(val_stats, epoch + 1)

                print(self.history)

                if early_stopping is not None:
                    valid_loss = val_stats.loss()
                    # early_stopping needs the validation loss to check if it has decreased,
                    # and if it has, it will make a checkpoint of the current model
                    if early_stopping(valid_loss):
                        self.history.log()
                        return
            else:
                print(self.history)
        self.history.log()


def predict(step, test_loader_step):
    print(f"Testing")
    with torch.no_grad():
        step.eval()
        stats = test_loader_step(step)
        metrics = Metrics(stats.outs(), stats.labels())
        print(metrics)
        metrics.log()
    return metrics()["Accuracy"]


class History:
    def __init__(self):
        self.history = {}
        self.epoch = 0
        self.timer = time.time()

    def __call__(self, stats, epoch):
        self.epoch = epoch

        if epoch in self.history:
            self.history[epoch].append(stats)
        else:
            self.history[epoch] = [stats]

    def __str__(self):
        epoch = f"\nEpoch {self.epoch};"
        stats = ' - '.join([f"{res}" for res in self.current()])
        timer = f"Time: {(time.time() - self.timer)}"

        return f"{epoch} - {stats} - {timer}"

    def current(self):
        return self.history[self.epoch]

    def log(self):
        msg = f"(Epoch: {self.epoch}) {' - '.join([f'({res})' for res in self.current()])}"
        logger.log_info("history", msg)
