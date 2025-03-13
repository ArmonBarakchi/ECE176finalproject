import os
import urllib.request
import tarfile
import zipfile
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
import glob
import utils
from utils import KITTIDataset, kitti_collate_fn
from InceptionNetwork import InceptionDetector, InceptionDetectionLoss, train_inception_model
from Network import KITTIObjectDetector, DetectionLoss, train_model
from torch.utils.data import random_split, Subset

# Select a random subset of the dataset

def main(Network):
    # Define KITTI dataset URL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    full_dataset = KITTIDataset(image_dir, label_dir)

    print(f"Total dataset size: {len(full_dataset)}")  # Debugging
    #Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


    # # Create a subset of your full dataset
    # subset_size = 500
    # indices = list(range(subset_size))
    # small_dataset = Subset(full_dataset, indices)
    # # Now split the small dataset into training and validation parts
    # train_dataset, val_dataset = random_split(small_dataset, [400, 100])



    # Create DataLoaders for both training and validation sets
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=kitti_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=kitti_collate_fn
    )


    if(Network == "ResNet18"):
        model = KITTIObjectDetector(num_classes=9)
        print("Training Resnet18")
        #   trained_model, train_acc, val_acc = train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001)
        # Load the trained model
        model = KITTIObjectDetector(num_classes=9)
        model.load_state_dict(
            torch.load("resnet18_kitti.pth", map_location=device))  # Update with correct checkpoint path
        model.to(device)
        model.eval()

        # List of test images
        test_images = [
            "./kitti_dataset/testing/image_2/000100.png",
            "./kitti_dataset/testing/image_2/000007.png",
        ]

        # Run inference and visualize results
        utils.predict_and_visualize(model, test_images)

if __name__ == "__main__":
    main(Network = "ResNet18")
