import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define the KITTIObjectDetector using a ResNet-18 backbone
class KITTIObjectDetector(nn.Module):
    def __init__(self, num_classes=9, backbone_name='resnet18', pretrained=True):
        super(KITTIObjectDetector, self).__init__()
        resnet = models.__dict__[backbone_name](pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = 512
        self.fc = nn.Linear(in_features, 256)
        self.detector = nn.Linear(256, num_classes + 4)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.detector(x)
        class_scores = x[:, :self.num_classes]
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
        batch_size = outputs['classes'].size(0)
        class_targets = []
        bbox_targets = []
        for i in range(batch_size):
            valid_indices = (targets['classes'][i] != -1).nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                class_targets.append(torch.tensor(0, device=device))
                bbox_targets.append(torch.zeros(4, device=device))
            else:
                idx = valid_indices[0]
                class_targets.append(targets['classes'][i][idx])
                bbox_targets.append(targets['bboxes'][i][idx])
        class_targets = torch.stack(class_targets)
        bbox_targets = torch.stack(bbox_targets)
        class_loss = self.class_loss_fn(outputs['classes'], class_targets)
        bbox_loss = self.bbox_loss_fn(outputs['bboxes'], bbox_targets)
        total_loss = class_loss + self.lambda_reg * bbox_loss
        return total_loss

# 3. Define the training function
def train_model(model, train_loader, val_loader, num_epochs=2, lr=0.001, save_path="resnet18_kitti.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = DetectionLoss().to(device)
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, train_correct, train_total = 0.0, 0, 0

        for images, targets in train_loader:
            images = images.to(device)
            targets['classes'] = targets['classes'].to(device)
            targets['bboxes'] = targets['bboxes'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            for i in range(images.size(0)):
                valid_indices = (targets['classes'][i] != -1).nonzero(as_tuple=True)[0]
                if len(valid_indices) == 0:
                    continue
                gt_class = targets['classes'][i][valid_indices[0]]
                pred_class = torch.argmax(outputs['classes'][i])
                if pred_class == gt_class:
                    train_correct += 1
                train_total += 1

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_accuracies.append(train_acc)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_acc * 100:.2f}%")

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets['classes'] = targets['classes'].to(device)
                targets['bboxes'] = targets['bboxes'].to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

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
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%")

    # Save the trained model
    print(f"Saving trained model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully!")

    return model, train_accuracies, val_accuracies
