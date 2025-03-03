
import os
import urllib.request
import zipfile
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np

#function to download KITTI
def download_kitti(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded {save_path}")
    else:
        print(f"File {save_path} already exists.")

# Function to extract KITTI dataset
def extract_kitti(file_path, extract_to):
    print(f"Extracting {file_path}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


class KITTIDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, img_size=(1280, 384)): #img_size is this way because cv2.resize takes wdith, height
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_size = img_size  # Fixed size for all images
        self.image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

        # Define KITTI classes for one-hot encoding
        self.classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                        'Cyclist', 'Tram', 'Misc', 'DontCare']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]

        # Load Image using OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize Image to a Fixed Size
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)

        # Normalize and Convert to Tensor
        image = image.astype(np.float32) / 255.0  # Normalize
        image = torch.tensor(image).permute(2, 0, 1)  # Convert HWC to CHW format

        # Load Labels
        with open(label_path, "r") as f:
            labels = f.readlines()

        # Extract object classes and convert to one-hot encoding
        objects = [line.split()[0] for line in labels]

        # Create one-hot encoded labels
        one_hot = torch.zeros(len(self.classes))
        for obj in objects:
            if obj in self.class_to_idx:
                one_hot[self.class_to_idx[obj]] = 1

        # Apply additional transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, one_hot
