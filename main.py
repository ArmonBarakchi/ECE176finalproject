import os
import urllib.request
import tarfile
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
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

# Function to download KITTI dataset

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
dataset = KITTIDataset(image_dir, label_dir)

# Create DataLoader for batching
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


