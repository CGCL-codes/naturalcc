from torch.utils.data import Dataset

class NccDataset(Dataset):
    def __init__(self,input):
        self.input = input
        
    def __len__(self):
        return len(self.input)
        
    def __getitem__(self,index):
        if index >= len(self.input):
            raise IndexError('Index out of range')
        sample = {
            'input': self.input[index]
        }
        return sample