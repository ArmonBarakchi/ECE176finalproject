

import urllib.request
import zipfile
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

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
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        self.label_files = sorted(glob.glob(os.path.join(self.label_dir, "*.txt")))
        self.transform = transform if transform else transforms.ToTensor()
        self.classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        image = self.transform(image)
        label_path = self.label_files[idx]
        with open(label_path, "r") as f:
            lines = f.readlines()
        class_indices = []
        bboxes = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 8 and parts[0] in self.class_to_idx:
                class_idx = self.class_to_idx[parts[0]]
                class_indices.append(class_idx)
                x1, y1, x2, y2 = map(float, parts[4:8])
                bbox = [x1/width, y1/height, x2/width, y2/height]
                bboxes.append(bbox)
        if len(class_indices) == 0:
            class_tensor = torch.zeros(0, dtype=torch.long)
            bbox_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            class_tensor = torch.tensor(class_indices, dtype=torch.long)
            bbox_tensor = torch.tensor(bboxes, dtype=torch.float32)
        targets = {"classes": class_tensor, "bboxes": bbox_tensor}
        return image, targets



import torch
import torch.nn.functional as F


def kitti_collate_fn(batch):
    """
    Custom collate function that pads images in the batch to the same size.
    Each item in the batch is a tuple (image, targets) where:
      - image is a tensor of shape (C, H, W)
      - targets is a dict with keys 'classes' and 'bboxes'
    """
    images, targets = zip(*batch)

    # Determine max height and width in the batch
    max_height = max([img.shape[1] for img in images])
    max_width = max([img.shape[2] for img in images])

    padded_images = []
    for img in images:
        _, h, w = img.shape
        pad_bottom = max_height - h
        pad_right = max_width - w
        # pad format: (pad_left, pad_right, pad_top, pad_bottom)
        # We assume no padding is needed for top and left.
        padded_img = F.pad(img, (0, pad_right, 0, pad_bottom), mode="constant", value=0)
        padded_images.append(padded_img)

    batched_images = torch.stack(padded_images, dim=0)

    # Keep targets as a list (variable number of objects per image)
    batched_targets = list(targets)

    return batched_images, batched_targets




def predict_and_visualize_all(model, image_paths, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), threshold=0.9):
    model.to(device)
    model.eval()
    transform = transforms.ToTensor()
    class_labels = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            tensor = transform(image)
            input_tensor = tensor.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            candidates = outputs['classes'][0]
            bboxes = outputs['bboxes'][0]
            scores = torch.softmax(candidates, dim=1)
            detections = []
            for i in range(candidates.size(0)):
                max_score, pred_class = torch.max(scores[i], dim=0)
                if max_score.item() < threshold:
                    continue
                detections.append((bboxes[i], pred_class.item(), max_score.item()))
            img_w, img_h = image.size
            image_np = tensor.mul(255).permute(1,2,0).byte().cpu().numpy()
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            for bbox, cls_idx, score in detections:
                x1, y1, x2, y2 = bbox.cpu().numpy() * np.array([img_w, img_h, img_w, img_h])
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(image_cv, f"{class_labels[cls_idx]} ({score:.2f})", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            plt.figure(figsize=(8,4))
            plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            plt.title("Predictions")
            plt.axis("off")
            plt.show()
