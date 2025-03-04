import os
import urllib.request
import tarfile
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import glob
import utils
from utils import KITTIDataset

# Define KITTI dataset URL
KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
ANNOTATION_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"

# Define dataset directory
DATASET_DIR = "./kitti_dataset"

# Download KITTI images and labels
os.makedirs(DATASET_DIR, exist_ok=True)
image_zip = os.path.join(DATASET_DIR, "kitti_images.zip")
label_zip = os.path.join(DATASET_DIR, "kitti_labels.zip")

utils.download_kitti(KITTI_URL, image_zip)
utils.download_kitti(ANNOTATION_URL, label_zip)

# Extract dataset
utils.extract_kitti(image_zip, DATASET_DIR)
utils.extract_kitti(label_zip, DATASET_DIR)

image_dir = os.path.join(DATASET_DIR, "training", "image_2")
label_dir = os.path.join(DATASET_DIR, "training", "label_2")

# Create dataset
full_dataset = KITTIDataset(image_dir, label_dir)

# Calculate split sizes
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# Split the dataset
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders for both training and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Print dataset sizes for verification
print(f"Total dataset size: {total_size}")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
