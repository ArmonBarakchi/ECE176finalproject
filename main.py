import os
import urllib.request
import tarfile
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import cv2
import numpy as np
import glob
import utils
from utils import KITTIDataset, kitti_collate_fn
from Network import KITTIObjectDetector, DetectionLoss, train_model

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
# utils.extract_kitti(image_zip, DATASET_DIR)
# utils.extract_kitti(label_zip, DATASET_DIR)

image_dir = os.path.join(DATASET_DIR, "training", "image_2")
label_dir = os.path.join(DATASET_DIR, "training", "label_2")

# Create dataset
full_dataset = KITTIDataset(image_dir, label_dir)

# Calculate split sizes
# total_size = len(full_dataset)
# print(total_size)
# train_size = int(0.8 * total_size)
# val_size = total_size - train_size
# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

subset_size = 1000

# Create a subset of your full dataset
indices = list(range(subset_size))
small_dataset = Subset(full_dataset, indices)

# Now split the small dataset into training and validation parts
train_dataset, val_dataset = random_split(small_dataset, [800, 200])
# Split the dataset


image,target = train_dataset[0]

# Create DataLoaders for both training and validation sets
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=kitti_collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=kitti_collate_fn
)


# Initialize model
model = KITTIObjectDetector(num_classes=9)
trained_model, train_acc, val_acc = train_model(model, train_loader, val_loader, num_epochs=2, lr=0.001)


