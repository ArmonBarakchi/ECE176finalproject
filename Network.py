# Network.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

# 1. Define the KITTIObjectDetector using a ResNet-18 backbone
class KITTIObjectDetector(nn.Module):
    def __init__(self, num_classes=9, backbone_name='resnet18', pretrained=True):
        super(KITTIObjectDetector, self).__init__()
        # Load a pre-trained ResNet model (now ResNet-18)
        resnet = models.__dict__[backbone_name](pretrained=pretrained)
        # Remove the fully connected layers and the average pool.
        # For ResNet-18, children()[:-2] retains features up to the last conv layer.
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        # Adaptive pooling to convert variable spatial dimensions to fixed (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # ResNet-18 outputs 512 channels (instead of 2048 for ResNet-50)
        in_features = 512
        # A fully connected layer to reduce feature dimensions
        self.fc = nn.Linear(in_features, 256)
        # Detector head: outputs num_classes raw scores and 4 bounding box coordinates
        self.detector = nn.Linear(256, num_classes + 4)
        self.num_classes = num_classes

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.feature_extractor(x)  # (B, 512, H', W')
        x = self.pool(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B, 512)
        x = F.relu(self.fc(x))  # (B, 256)
        x = self.detector(x)  # (B, num_classes + 4)
        # First num_classes are class scores (logits)
        class_scores = x[:, :self.num_classes]
        # Next 4 outputs are bbox coordinates, normalized to [0,1]
        bbox_preds = torch.sigmoid(x[:, self.num_classes:])
        return {'classes': class_scores, 'bboxes': bbox_preds}


# 2. Define a combined detection loss function
class DetectionLoss(nn.Module):
    def __init__(self, lambda_reg=1.0):
        super(DetectionLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.bbox_loss_fn = nn.SmoothL1Loss()

    def forward(self, outputs, targets):
        """
        outputs: dict with 'classes': (B, num_classes) and 'bboxes': (B, 4)
        targets: dict with 'classes': (B, max_boxes) and 'bboxes': (B, max_boxes, 4)
        For simplicity, we pick the first valid detection per image.
        """
        batch_size = outputs['classes'].size(0)
        class_targets = []
        bbox_targets = []
        for i in range(batch_size):
            # Valid detections are where class label != -1 (as padded in collate_fn)
            valid_indices = (targets['classes'][i] != -1).nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                # No valid detection: use a default value (class 0 and bbox zeros)
                class_targets.append(torch.tensor(0, device=outputs['classes'].device))
                bbox_targets.append(torch.zeros(4, device=outputs['classes'].device))
            else:
                idx = valid_indices[0]  # Use the first valid detection
                class_targets.append(targets['classes'][i][idx])
                bbox_targets.append(targets['bboxes'][i][idx])
        class_targets = torch.stack(class_targets)  # (B,)
        bbox_targets = torch.stack(bbox_targets)  # (B, 4)

        # Classification loss: note outputs['classes'] shape is (B, num_classes)
        class_loss = self.class_loss_fn(outputs['classes'], class_targets)
        # Regression loss for bounding boxes
        bbox_loss = self.bbox_loss_fn(outputs['bboxes'], bbox_targets)
        total_loss = class_loss + self.lambda_reg * bbox_loss
        return total_loss


# 3. Define the training function
def train_model(model, train_loader, val_loader, num_epochs=2, lr=0.001, device=torch.device('cpu')):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    criterion = DetectionLoss()

    # Lists to store accuracy for each epoch
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images = images.to(device)
            # Move target tensors to device
            targets['classes'] = targets['classes'].to(device)
            targets['bboxes'] = targets['bboxes'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            # Compute training accuracy for this batch
            # For each image, pick the first valid target (where label != -1)
            for i in range(images.size(0)):
                valid_indices = (targets['classes'][i] != -1).nonzero(as_tuple=True)[0]
                if len(valid_indices) == 0:
                    continue  # Skip if no valid target
                gt_class = targets['classes'][i][valid_indices[0]]
                pred_class = torch.argmax(outputs['classes'][i])
                if pred_class == gt_class:
                    train_correct += 1
                train_total += 1

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_accuracies.append(train_acc)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_acc * 100:.2f}%")

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets['classes'] = targets['classes'].to(device)
                targets['bboxes'] = targets['bboxes'].to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

                # Compute validation accuracy for this batch
                for i in range(images.size(0)):
                    valid_indices = (targets['classes'][i] != -1).nonzero(as_tuple=True)[0]
                    if len(valid_indices) == 0:
                        continue
                    gt_class = targets['classes'][i][valid_indices[0]]
                    pred_class = torch.argmax(outputs['classes'][i])
                    if pred_class == gt_class:
                        val_correct += 1
                    val_total += 1

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_accuracies.append(val_acc)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%")

    return model, train_accuracies, val_accuracies
