from ..utils.objects import stats


class LoaderStep:
    def __init__(self, name, data_loader, device):
        self.name = name
        self.loader = data_loader
        self.size = len(data_loader)
        self.device = device

    def __call__(self, step):
        self.stats = stats.Stats(self.name)

        for i, batch in enumerate(self.loader):
            batch.to(self.device)
            stat: stats.Stat = step(i, batch, batch.y)
            self.stats(stat)

        return self.stats
