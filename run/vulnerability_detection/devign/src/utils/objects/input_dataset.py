from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import DataLoader


class InputDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index].input

    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)
