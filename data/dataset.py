import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

def load_data(data_dir):
    data = {'imgpath': [], 'labels': []}
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and (file.endswith('.jpg') or file.endswith('.png')):
                    data['imgpath'].append(file_path)
                    data['labels'].append(folder)
    return pd.DataFrame(data)

class HeadgearDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 2]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
