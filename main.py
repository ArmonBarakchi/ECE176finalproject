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
from bbox import eval, save_detection_image
import matplotlib.pyplot as plt

def main(Network):
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

    #Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])




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


    if(Network == "ResNet18"):
        model = KITTIObjectDetector(num_classes=9)
        print("Training Resnet18")
        trained_model, train_acc, val_acc = train_model(model, train_loader, val_loader, num_epochs=2, lr=0.001)
    if(Network == "Inception"):

        model = KITTIObjectDetector(num_classes=9)
        print("Training Inception")
        trained_model, train_acc, val_acc = train_inception_model(model, train_loader, val_loader, num_epochs=2, lr=2e-3)
    
    torch.save(trained_model.state_dict(), 'resnet18_kitti_detector.pth')
    print("Visualizing...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval(trained_model, val_loader, num_samples=5, threshold=0.3, device=device)

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{Network} Training and Validation Accuracy')
    plt.legend()
    plt.savefig(f'{Network}_accuracy_plot.png')
    plt.show()
    
    # Test on a single image
    print("Testing on a single image...")
    # Get a sample image from the validation set
    sample_idx = 0
    sample_image, sample_target = val_dataset[sample_idx]
    
    # Save the sample image
    sample_image_path = f'{Network}_sample_image.png'
    plt.imsave(sample_image_path, sample_image.permute(1, 2, 0).numpy())
    
    # Run detection and save the result
    detection_path = save_detection_image(sample_image_path, trained_model, 
                                         output_path=f'{Network}_detection_result.png', 
                                         threshold=0.3, device=device)
    print(f"Detection result saved to {detection_path}")

if __name__ == "__main__":
    main(Network = "Inception")
