import json
import os
import cv2
import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer

class BasketDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.video_paths = self._load_video_paths()

    def _load_video_paths(self):
        video_paths = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for video_file in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_file)
                video_paths.append((video_path, self.class_to_idx[cls]))
        return video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        label = torch.empty((1), dtype=torch.float)
        video_path, label[0] = self.video_paths[idx]

        # Reading of the JSON file
        with open(video_path, 'r') as file:
            data = json.load(file)
        
        # Put the values in a tensor
        values = list(data.values())
        sample = {'emb_fea': torch.tensor(values), 'label': label}
        return sample

def loadDataset(verbose=False):
    dataset = BasketDataset(root_dir='json')

    # Iterate through frames and display them
    if verbose: 
        # Output frames of one video from the dataset (change the index as needed)
        sample = dataset[0]
        feature = sample['emb_fea']
        label = sample['label']
        print(feature, label)
    return dataset