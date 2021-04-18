from ncc.data.wrappers.base_wrapper_dataset import BaseWrapperDataset


class SliceDataset(BaseWrapperDataset):
    """
    ```portion``` ahead part of dataset
    """

    def __init__(self, dataset, start=0, end=None):
        super().__init__(dataset)
        self.dataset = dataset
        self.start = start
        if end is None:
            self.end = len(dataset)
        elif end < 0:
            self.end = len(dataset) + end
        else:
            self.end = min(len(dataset), end)
        assert self.start < self.end, IndexError(f"start[{self.start}] should be greater then end[{self.end}]")

    def __getitem__(self, index):
        assert self.start <= index < self.end, IndexError(f"{self.start} <= {index} < {self.end}")
        item = self.dataset[index]
        return item

    def __len__(self):
        return self.end - self.start

    @property
    def sizes(self):
        return self.dataset.sizes[self.start:self.end]
