from ncc.data.wrappers.base_wrapper_dataset import BaseWrapperDataset


class PortionDataset(BaseWrapperDataset):
    """
    ```portion``` ahead part of dataset
    """

    def __init__(self, dataset, portion):
        super().__init__(dataset)
        assert portion is not None
        self.portion = portion
        self.dataset = dataset

    def __getitem__(self, index):
        assert index < len(self), (index, len(self))
        item = self.dataset[index]
        return item

    def __len__(self):
        return int(len(self.dataset) * self.portion)

    @property
    def sizes(self):
        return self.dataset.sizes[:len(self)]
