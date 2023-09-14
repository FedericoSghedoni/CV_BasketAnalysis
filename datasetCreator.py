import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as t

# Image Patching in tensor
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target

class BasketDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.video_paths = self._load_video_paths()
        self.transform = transform

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
        video_path, label = self.video_paths[idx]

        # Capture video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        sample = {'frames': frames, 'label': label}

        return sample

def loadDataset(verbose=False):
    transform = [t.ToPILImage(), t.Resize((144, 144)), t.ToTensor()]
    dataset = BasketDataset(root_dir='dataset', transform=t.Compose(transform))

    # Iterate through frames and display them
    if verbose: 
        # Output frames of one video from the dataset (change the index as needed)
        sample = dataset[0]
        frames = sample['frames']
        for frame in frames:
            frame = torch.reshape(frame,(720,720,3))
            cv2.imshow('Video Frame', frame.numpy())
            cv2.waitKey(50)  # Adjust the delay as needed (ms)
        cv2.destroyAllWindows()
    return dataset