

import urllib.request
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torchvision.transforms import ToPILImage, ToTensor
import glob
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, ConcatDataset, Dataset
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

#container for night data
class NightDataset(Dataset):

    def __init__(self, base_dataset, gamma=2.0, blue_tint=30):
        self.base_dataset = base_dataset
        self.gamma = gamma
        self.blue_tint = blue_tint
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

    # applies night transformation to data
    def simulate_night(self, image):

        img_np = np.array(image).astype(np.float32)
        # Apply gamma correction: out = 255 * (in/255)^gamma
        img_np = 255.0 * np.power(img_np / 255.0, self.gamma)
        # Add blue tint: reduce red and green channels, and increase blue channel.
        img_np[..., 0] *= 0.8  # reduce red channel
        img_np[..., 1] *= 0.8  # reduce green channel
        img_np[..., 2] = np.clip(img_np[..., 2] + self.blue_tint, 0, 255)  # increase blue channel
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        # If the image is a tensor, convert it to a PIL image.
        if not isinstance(image, Image.Image):
            image = self.to_pil(image)
        # Apply the nighttime simulation.
        night_image = self.simulate_night(image)
        night_image = self.to_tensor(night_image)
        return night_image, target

#container for Noisy data
class NoisyDataset(Dataset):

    def __init__(self, base_dataset, noise_mean=0.0, noise_std=0.1):

        self.base_dataset = base_dataset
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        # Generate Gaussian noise with the same shape as the image.
        noise = torch.randn_like(image) * self.noise_std + self.noise_mean
        noisy_image = torch.clamp(image + noise, 0.0, 1.0)
        return noisy_image, target

#container for dataset
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

#ensures data is ready to go into the network despite variable image dimensions
def kitti_collate_fn(batch):
    images, targets = zip(*batch)

    # Determine max height and width in the batch
    max_height = max([img.shape[1] for img in images])
    max_width = max([img.shape[2] for img in images])

    padded_images = []
    for img in images:
        _, h, w = img.shape
        pad_bottom = max_height - h
        pad_right = max_width - w
        padded_img = F.pad(img, (0, pad_right, 0, pad_bottom), mode="constant", value=0)
        padded_images.append(padded_img)

    batched_images = torch.stack(padded_images, dim=0)

    batched_targets = list(targets)

    return batched_images, batched_targets

#will only draw the highest probability bounding box
def predict_and_visualize(model, image_paths, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                              threshold=0.99):
    model.to(device)
    model.eval()
    transform = transforms.ToTensor()
    # "DontCare" is the last class in this list; we will ignore detections predicted as "DontCare"
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
                # Skip candidate if its score is below the threshold
                if max_score.item() < threshold:
                    continue
                # Skip candidate if predicted as "DontCare"
                if pred_class.item() == len(class_labels) - 1:
                    continue
                detections.append((bboxes[i], pred_class.item(), max_score.item()))

            img_w, img_h = image.size
            image_np = tensor.mul(255).permute(1, 2, 0).byte().cpu().numpy()
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Only draw one bounding box: choose the detection with the highest score
            if detections:
                best_bbox, best_cls_idx, best_score = max(detections, key=lambda d: d[2])
                x1, y1, x2, y2 = best_bbox.cpu().numpy() * np.array([img_w, img_h, img_w, img_h])
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_cv, f"{class_labels[best_cls_idx]} ({best_score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            plt.figure(figsize=(8, 4))
            plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            plt.title("Predictions")
            plt.axis("off")
            plt.show()

#makes duplicates of images and draws all the bounding boxes
def predict_and_visualize2(model, image_paths, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                              threshold=0.5):
        model.to(device)
        model.eval()
        transform = transforms.ToTensor()
        # "DontCare" is the last class in this list; we will ignore detections predicted as "DontCare"
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
                    # Skip candidate if its score is below the threshold
                    if max_score.item() < threshold:
                        continue
                    # Skip candidate if predicted as "DontCare"
                    if pred_class.item() == len(class_labels) - 1:
                        continue
                    detections.append((bboxes[i], pred_class.item(), max_score.item()))

                img_w, img_h = image.size
                image_np = tensor.mul(255).permute(1, 2, 0).byte().cpu().numpy()
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # Only draw one bounding box: choose the detection with the highest score
                # For each detection, draw the bounding box on a separate copy of the image
                if detections:
                    for bbox, cls_idx, score in detections:
                        image_copy = image_cv.copy()
                        x1, y1, x2, y2 = bbox.cpu().numpy() * np.array([img_w, img_h, img_w, img_h])
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image_copy, f"{class_labels[cls_idx]} ({score:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        plt.figure(figsize=(8, 4))
                        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
                        plt.title("Predictions")
                        plt.axis("off")
                        plt.show()

                plt.figure(figsize=(8, 4))
                plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
                plt.title("Predictions")
                plt.axis("off")
                plt.show()

#splits dataset into half noise half clean
def split_half_noisy(dataset, noise_mean=0.0, noise_std=0.05):

    total_indices = list(range(len(dataset)))
    split_index = len(dataset) // 2
    noisy_indices = total_indices[:split_index]
    clean_indices = total_indices[split_index:]

    noisy_subset = Subset(dataset, noisy_indices)
    clean_subset = Subset(dataset, clean_indices)

    # Wrap only the noisy subset with NoisyDataset
    noisy_subset = NoisyDataset(noisy_subset, noise_mean=noise_mean, noise_std=noise_std)

    # Concatenate the noisy and clean subsets back together
    combined_dataset = ConcatDataset([noisy_subset, clean_subset])
    return combined_dataset