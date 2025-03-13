

import urllib.request
import zipfile
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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
    def __init__(self, img_dir, label_dir, cache_dir='./cache', img_size=(640, 192), transform=None):
        """
        - img_size: Reduced image size for faster processing.
        - cache_dir: Directory where preprocessed images are saved.
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.img_size = img_size

        self.image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

        # Use torchvision transforms for efficient operations
        self.transform = transform or transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),  # Converts and scales pixel values to [0,1]
        ])

        # Define KITTI classes for one-hot encoding
        self.classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                        'Cyclist', 'Tram', 'Misc', 'DontCare']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def _cache_path(self, img_path):
        # Create a unique name for the cached version of the image
        base = os.path.basename(img_path)
        return os.path.join(self.cache_dir, base.replace('.png', '.npy'))

    def _load_image(self, img_path):
        # Check if cached version exists
        cache_path = self._cache_path(img_path)
        if os.path.exists(cache_path):
            image = np.load(cache_path)
            # Convert to torch tensor with appropriate shape: (C, H, W)
            return torch.tensor(image, dtype=torch.float32)
        else:
            # Open with PIL for faster transforms
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)  # (C, H, W)
            # Save the processed image to cache for future runs
            np.save(cache_path, image.numpy())
            return image

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]

        # Load image (from cache if available)
        image = self._load_image(img_path)

        # Load and process labels
        with open(label_path, "r") as f:
            labels = f.readlines()

        class_indices = []
        bboxes = []
        for line in labels:
            parts = line.split()
            if len(parts) >= 8 and parts[0] in self.class_to_idx:
                class_idx = self.class_to_idx[parts[0]]
                class_indices.append(class_idx)

                # Extract bounding box and normalize according to resized dimensions
                x1, y1, x2, y2 = map(float, parts[4:8])
                # Since image was resized, you may normalize with new dimensions (width, height)
                bbox = [x1 / self.img_size[0], y1 / self.img_size[1],
                        x2 / self.img_size[0], y2 / self.img_size[1]]
                bboxes.append(bbox)

        targets = {
            'classes': torch.tensor(class_indices, dtype=torch.long),
            'bboxes': torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((1, 4), dtype=torch.float32)
        }
        return image, targets


def kitti_collate_fn(batch):
    """
    Custom collate function to handle variable-length object detection labels.

    Args:
        batch: List of (image, targets) tuples from the dataset.

    Returns:
        Tuple of (batched_images, batched_targets) with consistent dimensions.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images into a tensor
    batched_images = torch.stack(images, dim=0)

    # Determine max number of bounding boxes in a batch
    max_bboxes = max(len(t['bboxes']) for t in targets)

    # Create tensors for class labels and bounding boxes
    class_tensors = []
    bbox_tensors = []

    for target in targets:
        cls_tensor = target['classes']
        bbox_tensor = target['bboxes']

        # Pad class labels with -1 (meaning "no object" for extra padding)
        cls_padded = torch.full((max_bboxes,), -1, dtype=torch.long)
        cls_padded[:len(cls_tensor)] = cls_tensor  # Fill with actual class data

        # Pad bounding boxes with zeros
        bbox_padded = torch.zeros((max_bboxes, 4), dtype=torch.float32)
        bbox_padded[:len(bbox_tensor)] = bbox_tensor  # Fill with actual bbox data

        class_tensors.append(cls_padded)
        bbox_tensors.append(bbox_padded)

    # Stack everything into final batch tensors
    batched_targets = {
        'classes': torch.stack(class_tensors, dim=0),  # (batch, max_boxes)
        'bboxes': torch.stack(bbox_tensors, dim=0)    # (batch, max_boxes, 4)
    }

    return batched_images, batched_targets

import matplotlib.pyplot as plt
import cv2

def predict_and_visualize(model, image_paths, img_size=(640, 192), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Loads images, runs inference with the trained model, and visualizes bounding box predictions.

    Args:
        model (KITTIObjectDetector): Trained model.
        image_paths (list of str): List of image file paths.
        img_size (tuple): Resized image dimensions (Width, Height).
        device (torch.device): CUDA or CPU.
    """
    # Ensure model is in evaluation mode
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    class_labels = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                    'Cyclist', 'Tram', 'Misc', 'DontCare']

    with torch.no_grad():
        for image_path in image_paths:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

            # Run inference
            outputs = model(input_tensor)

            # Extract class scores and bounding boxes
            class_scores = outputs['classes'].squeeze(0)  # (num_classes,)
            bbox_preds = outputs['bboxes'].squeeze(0)  # (4,)

            # Get the predicted class
            pred_class_idx = torch.argmax(class_scores).item()
            pred_class = class_labels[pred_class_idx]
            pred_score = torch.softmax(class_scores, dim=0)[pred_class_idx].item()

            # Convert bbox from normalized (0-1) to actual image coordinates
            img_w, img_h = image.size
            x1, y1, x2, y2 = bbox_preds.cpu().numpy() * np.array([img_w, img_h, img_w, img_h])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Display results
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv, f"{pred_class} ({pred_score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert back to RGB for Matplotlib
            plt.figure(figsize=(8, 4))
            plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            plt.title(f"Prediction: {pred_class} ({pred_score:.2f})")
            plt.axis("off")
            plt.show()
