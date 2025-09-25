from typing import Dict, Any
from PIL import Image
from abc import ABC, abstractmethod
import sys, os
import datasets

class BaseDataset(ABC):
    """
    A base dataset class with items in the format: {'image': PIL.Image, 'text': str}.
    Subclasses should implement their own data loading methods and properties.
    """

    def __init__(self, range_ids:tuple=None):
        self.data = []  # List to store dataset items as dictionaries
        self.range_ids = range_ids
        self.load_data()

    @abstractmethod
    def load_data(self):
        """
        Abstract method for loading data. Subclasses must implement this method.
        """
        pass

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get an item by index.

        :param index: Index of the desired item.
        :return: A dictionary with 'image' (PIL.Image) and 'text' (str).
        """
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of range")
        return self.data[index]

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        :return: Number of items in the dataset.
        """
        return len(self.data)

            
# Design2code-hard dataset, 50 samples for test
class D2CHardDataset(BaseDataset):
    def load_data(self):   
        self.data_path = 'data/Design2Code-HARD'
        for i in range(len(os.listdir(self.data_path))): 
            item = {}
            image_path = os.path.join(self.data_path, f'g{i}.png')
            html_path = os.path.join(self.data_path, f'g{i}.html')
            
            if not (os.path.exists(image_path) and os.path.exists(html_path)):
                continue
            
            item['image'] = Image.open(image_path)
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    item['text'] = f.read()  # HTML
            
            self.data.append(item)
        if self.range_ids:
            self.range_ids[1] = self.range_ids[1] if self.range_ids[1]>0 else len(self.data)
            self.data = self.data[self.range_ids[0]:self.range_ids[1] ]   
            
# CC-HARD dataset
class V2UDataset(BaseDataset):
     def load_data(self):   
        self.data_path = 'data/CC-HARD.parquet'
        self.data = datasets.load_dataset('parquet', data_files=self.data_path)['train']
        if self.range_ids:
            self.range_ids[1] = self.range_ids[1] if self.range_ids[1]>0 else len(self.data)
            self.data = self.data.select(range(self.range_ids[0],self.range_ids[1]))    
        
# CC-HARD dataset
class V2UDataset_old(BaseDataset):
     def load_data(self):   
        self.data_path = 'data/CC-HARD_old.parquet'
        self.data = datasets.load_dataset('parquet', data_files=self.data_path)['train']
        if self.range_ids:
            self.range_ids[1] = self.range_ids[1] if self.range_ids[1]>0 else len(self.data)
            self.data = self.data.select(range(self.range_ids[0],self.range_ids[1]))    


# Design2code-hard dataset, 50 samples for test
class tmpDataset(BaseDataset):
    def load_data(self):   
        self.data_path = 'data/tmp'
        for i in range(len(os.listdir(self.data_path))): 
            item = {}
            image_path = os.path.join(self.data_path, f'g{i}.png')
            html_path = os.path.join(self.data_path, f'g{i}.html')
            
            if not (os.path.exists(image_path) and os.path.exists(html_path)):
                continue
            
            item['image'] = Image.open(image_path)
            if os.path.exists(html_path):
                with open(html_path, 'r') as f:
                    item['text'] = f.read()  # HTML
            
            self.data.append(item)
        if self.range_ids:
            self.range_ids[1] = self.range_ids[1] if self.range_ids[1]>0 else len(self.data)
            self.data = self.data[self.range_ids[0]:self.range_ids[1] ]   
    
# Example usage
if __name__ == "__main__":
    ds = V2UDataset()
    print(ds[0])
