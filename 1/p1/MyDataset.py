import os
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, annotation_path, ):
        super(MyDataset, self).__init__()
        img_path = []
        img_label = []
        
        pass

    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)
    
    