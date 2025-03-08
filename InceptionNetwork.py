# InceptionNetwork.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

# 1. Define the KITTIObjectDetector using an Inception backbone
class InceptionDetector(nn.Module):
    def __init__(self, num_classes=9, pretrained=True):
        super(InceptionDetector, self).__init__()
        # Load a pre-trained Inception v3 model
        inception = models.inception_v3(pretrained=pretrained)
        # Remove the fully connected layers and the average pool
        # For Inception, we need to keep the features up to the last Mixed_7c layer
        self.feature_extractor = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c
        )
        # Adaptive pooling to convert variable spatial dimensions to fixed (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Inception v3 outputs 2048 channels at the last Mixed_7c layer
        in_features = 2048
        # A fully connected layer to reduce feature dimensions
        self.fc = nn.Linear(in_features, 256)
        # Detector head: outputs num_classes raw scores and 4 bounding box coordinates
        self.detector = nn.Linear(256, num_classes + 4)
        self.num_classes = num_classes

    def forward(self, x):
        # Inception v3 expects input size of 299x299, but we'll use adaptive pooling
        # to handle different input sizes
        # x: (B, 3, H, W)
        x = self.feature_extractor(x)  # (B, 2048, H', W')
        x = self.pool(x)  # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B, 2048)
        x = F.relu(self.fc(x))  # (B, 256)
        x = self.detector(x)  # (B, num_classes + 4)
        # First num_classes are class scores (logits)
        class_scores = x[:, :self.num_classes]
        # Next 4 outputs are bbox coordinates, normalized to [0,1]
        bbox_preds = torch.sigmoid(x[:, self.num_classes:])
        return {'classes': class_scores, 'bboxes': bbox_preds}


# 2. Define a combined detection loss function
class InceptionDetectionLoss(nn.Module):
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
def train_inception_model(model, train_loader, val_loader, num_epochs=2, lr=0.001, device=torch.device('cpu')):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
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