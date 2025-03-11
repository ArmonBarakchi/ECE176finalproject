
import urllib.request
import zipfile
import matplotlib.pyplot as plt
import cv2
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


import cv2
import numpy as np


def draw_bboxes(image, bboxes, labels, class_names):
    """
    Draws bounding boxes and class labels on the image.
    """
    print(f"\n[DEBUG] Drawing BBoxes: {bboxes}")

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = map(int, bbox)

        # Ensure box has a valid size
        if x2 > x1 and y2 > y1:
            print(f"[DEBUG] Drawing bbox {bbox} for class {class_names[label]}")
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            class_name = class_names[label] if label < len(class_names) else "Unknown"
            cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        else:
            print(f"[DEBUG] Skipping invalid bbox: {bbox}")

    return image



import random

def visualize_predictions(model, val_loader, class_names, num_images=5, device="cpu"):
    """
    Runs inference on a shuffled batch of validation images and draws predicted bounding boxes.
    """
    model.eval()
    model.to(device)

    # Get a random batch from val_loader
    val_iter = iter(val_loader)
    images, targets = next(val_iter)  # Fetch a batch

    batch_size = images.shape[0]
    num_images = min(num_images, batch_size)  # Ensure we don't exceed batch size
    selected_indices = random.sample(range(batch_size), num_images)
    images = images[selected_indices].to(device)

    # Shuffle the original dataset list before picking images
    image_dir = "./kitti_dataset/training/image_2"
    original_image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    random.shuffle(original_image_paths)  # Ensures new images are chosen each time
    selected_image_paths = [original_image_paths[i] for i in selected_indices]

    # Get original image sizes
    original_sizes = []
    for img_path in selected_image_paths:
        with Image.open(img_path) as img:
            original_sizes.append(img.size)  # (width, height)

    with torch.no_grad():
        outputs = model(images)  # Run inference

    for i, img_idx in enumerate(selected_indices):
        original_img = Image.open(selected_image_paths[i])
        orig_w, orig_h = original_sizes[i]

        # Extract class predictions
        pred_classes = torch.argmax(outputs["classes"][i], dim=0).cpu().numpy()
        print(f"[DEBUG] Image {i}: Predicted Class Scores: {outputs['classes'][i].cpu().numpy()}")
        print(f"[DEBUG] Selected Class: {class_names[pred_classes]}")

        # Extract bounding boxes
        pred_bboxes = outputs["bboxes"][i].cpu().numpy()
        print(f"[DEBUG] Raw Bounding Boxes from Model: {pred_bboxes}")

        if pred_bboxes.ndim == 1 and pred_bboxes.shape[0] == 4:
            pred_bboxes = pred_bboxes.reshape(1, 4)
        elif pred_bboxes.ndim == 0:
            pred_bboxes = np.zeros((1, 4))

        # Scale bounding boxes back to original image size
        scaled_bboxes = []
        for bbox in pred_bboxes:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(int(x1 * orig_w), orig_w - 1))
            y1 = max(0, min(int(y1 * orig_h), orig_h - 1))
            x2 = max(0, min(int(x2 * orig_w), orig_w - 1))
            y2 = max(0, min(int(y2 * orig_h), orig_h - 1))

            if x2 > x1 and y2 > y1 and (x2 - x1) > 5 and (y2 - y1) > 5:
                scaled_bboxes.append([x1, y1, x2, y2])
            else:
                print(f"[DEBUG] Skipping invalid bbox after scaling: {[x1, y1, x2, y2]}")

        scaled_bboxes = np.array(scaled_bboxes)
        print(f"[DEBUG] Corrected Scaled Bounding Boxes: {scaled_bboxes}")

        # Convert image for OpenCV
        original_img = np.array(original_img)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

        # Draw bounding boxes on the original image
        img_with_bboxes = draw_bboxes(original_img, scaled_bboxes, [pred_classes], class_names)

        # Show image with bounding boxes
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(img_with_bboxes, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Predicted: {class_names[pred_classes]}")
        plt.show()