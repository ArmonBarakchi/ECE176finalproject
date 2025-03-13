import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class KITTIObjectDetector(nn.Module):
    def __init__(self, num_classes=9, num_detections=5, backbone_name='resnet18', pretrained=True):
        """
        num_detections: number of candidate detections per image.
        The detector head outputs a tensor of shape (B, num_detections*(num_classes+4)).
        """
        super(KITTIObjectDetector, self).__init__()
        self.num_detections = num_detections
        self.num_classes = num_classes
        resnet = models.__dict__[backbone_name](pretrained=pretrained)
        # Use all layers except the final classification layers
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = 512
        self.fc = nn.Linear(in_features, 256)
        # Final detector outputs (num_classes + 4) scores for each candidate detection.
        self.detector = nn.Linear(256, num_detections * (num_classes + 4))

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.detector(x)  # shape: (B, num_detections*(num_classes+4))
        # Reshape to (B, num_detections, num_classes+4)
        x = x.view(x.size(0), self.num_detections, self.num_classes + 4)
        # Split into class scores and bounding box predictions.
        class_scores = x[:, :, :self.num_classes]  # (B, num_detections, num_classes)
        # For bounding boxes, use sigmoid to constrain them between 0 and 1 (assuming normalized coordinates)
        bbox_preds = torch.sigmoid(x[:, :, self.num_classes:])  # (B, num_detections, 4)
        return {'classes': class_scores, 'bboxes': bbox_preds}

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(areaA + areaB - interArea + 1e-6)
    return iou

def DetectionLoss(lambda_reg=1000.0, penalty=100.0):
    class _DetectionLoss(nn.Module):
        def __init__(self, lambda_reg, penalty):
            super(_DetectionLoss, self).__init__()
            self.lambda_reg = lambda_reg
            self.penalty = penalty
            self.class_loss_fn = nn.CrossEntropyLoss()
            self.bbox_loss_fn = nn.SmoothL1Loss()
        def forward(self, outputs, target):
            device = outputs['classes'].device
            gt_classes = target['classes']
            gt_bboxes = target['bboxes']
            N = outputs['classes'].size(0)
            total_loss = 0.0
            matched = 0
            for j in range(gt_classes.size(0)):
                gt_cls = gt_classes[j].item()
                gt_box = gt_bboxes[j]
                best_loss = None
                for i in range(N):
                    pred_cls_logits = outputs['classes'][i]
                    cls_logits = pred_cls_logits.view(1, -1)
                    target_cls = torch.tensor([gt_cls], device=device, dtype=torch.long).view(-1)
                    cls_loss = self.class_loss_fn(cls_logits, target_cls)
                    bbox_loss = self.bbox_loss_fn(outputs['bboxes'][i].view(1, -1), gt_box.view(1, -1))
                    extra = self.penalty if torch.argmax(pred_cls_logits).item() != gt_cls else 0.0
                    candidate_loss = cls_loss + self.lambda_reg * bbox_loss + extra
                    if best_loss is None or candidate_loss < best_loss:
                        best_loss = candidate_loss
                if best_loss is not None:
                    total_loss += best_loss
                    matched += 1
            if matched > 0:
                total_loss = total_loss / matched
            else:
                total_loss = torch.tensor(0.0, device=device) + 0.0 * outputs['classes'].sum()
            return total_loss
    return _DetectionLoss(lambda_reg, penalty)



def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, save_path="resnet18_kitti.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = DetectionLoss(lambda_reg=1.0, penalty=100.0).to(device)
    debug = True
    train_det_accs = []
    train_cls_accs = []
    val_det_accs = []
    val_cls_accs = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_gt_train = 0
        correct_det_train = 0
        correct_cls_train = 0
        batch_count = 0
        for images, targets in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            B = images.size(0)
            batch_loss = 0.0
            for i in range(B):
                target_i = {k: v.to(device) for k, v in targets[i].items()}
                output_i = {'classes': outputs['classes'][i], 'bboxes': outputs['bboxes'][i]}
                loss_i = criterion(output_i, target_i)
                batch_loss += loss_i
                gt_classes = target_i['classes']
                gt_bboxes = target_i['bboxes']
                num_gt = gt_classes.size(0)
                total_gt_train += num_gt
                for j in range(num_gt):
                    gt_cls = gt_classes[j].item()
                    gt_box = gt_bboxes[j].detach().cpu().numpy()
                    best_iou = 0.0
                    correct_class = False
                    num_dets = output_i['bboxes'].size(0)
                    for k in range(num_dets):
                        pred_cls = torch.argmax(output_i['classes'][k]).item()
                        if pred_cls == gt_cls:
                            correct_class = True
                        if pred_cls != gt_cls:
                            continue
                        cand_box = output_i['bboxes'][k].detach().cpu().numpy()
                        iou = compute_iou(cand_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                    if correct_class:
                        correct_cls_train += 1
                    if best_iou > 0.4:
                        correct_det_train += 1
            batch_loss = batch_loss / B
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item() * B
            batch_count += B
        epoch_loss = running_loss / batch_count
        train_det_acc = correct_det_train / total_gt_train if total_gt_train > 0 else 0.0
        train_cls_acc = correct_cls_train / total_gt_train if total_gt_train > 0 else 0.0
        print("Epoch {}/{} - Train Loss: {:.4f}, Train Det Acc: {:.2f}%, Train Cls Acc: {:.2f}%".format(
            epoch+1, num_epochs, epoch_loss, train_det_acc*100, train_cls_acc*100))
        train_det_accs.append(train_det_acc)
        train_cls_accs.append(train_cls_acc)
        model.eval()
        running_val_loss = 0.0
        total_gt_val = 0
        correct_det_val = 0
        correct_cls_val = 0
        val_batch_count = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                outputs = model(images)
                B = images.size(0)
                for i in range(B):
                    target_i = {k: v.to(device) for k, v in targets[i].items()}
                    output_i = {'classes': outputs['classes'][i], 'bboxes': outputs['bboxes'][i]}
                    loss_i = criterion(output_i, target_i)
                    running_val_loss += loss_i.item()
                    gt_classes = target_i['classes']
                    gt_bboxes = target_i['bboxes']
                    num_gt = gt_classes.size(0)
                    total_gt_val += num_gt
                    for j in range(num_gt):
                        gt_cls = gt_classes[j].item()
                        gt_box = gt_bboxes[j].detach().cpu().numpy()
                        best_iou = 0.0
                        correct_class = False
                        num_dets = output_i['bboxes'].size(0)
                        for k in range(num_dets):
                            pred_cls = torch.argmax(output_i['classes'][k]).item()
                            if pred_cls == gt_cls:
                                correct_class = True
                            if pred_cls != gt_cls:
                                continue
                            cand_box = output_i['bboxes'][k].detach().cpu().numpy()
                            iou = compute_iou(cand_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                        if correct_class:
                            correct_cls_val += 1
                        if best_iou > 0.5:
                            correct_det_val += 1
                val_batch_count += B
            val_loss_epoch = running_val_loss / val_batch_count
            val_det_acc = correct_det_val / total_gt_val if total_gt_val > 0 else 0.0
            val_cls_acc = correct_cls_val / total_gt_val if total_gt_val > 0 else 0.0
            print("Epoch {}/{} - Val Loss: {:.4f}, Val Det Acc: {:.2f}%, Val Cls Acc: {:.2f}%".format(
                epoch+1, num_epochs, val_loss_epoch, val_det_acc*100, val_cls_acc*100))
            val_det_accs.append(val_det_acc)
            val_cls_accs.append(val_cls_acc)
    torch.save(model.state_dict(), save_path)
    print("Model saved to {}".format(save_path))
    return model, train_det_accs, train_cls_accs, val_det_accs, val_cls_accs
