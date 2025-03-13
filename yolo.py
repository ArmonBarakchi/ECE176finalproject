import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models 
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import KITTI_DATASET, kitti_collate_fn


class YOLO(nn.Module):
    def __init__(self, num_classes, anchors, img_dim=416):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.img_dim = img_dim
        self.num_anchors = len(anchors)
        self.grid_size = 0
    
    def forward(self, x):
        batch_size = x.size(0)
        grid_size = s.size(2)
        pred = x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(pred[..., 0])
        y = torch.sigmoid(pred[..., 1])
        w = pred[..., 2]
        h = pred[..., 3]
        conf = torch.sigmoid(pred[..., 4])
        class_pred = torch.sigmoid(pred[..., 5:]) 

        if grid_size != self.grid_size: 
            self.grid_size = grid_size 
            self.stride = self.img_dim / self.grid_size
            grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).float()
            grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).float()
            scaled_anchors = torch.FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
            anchor_w = scaled_anchors[:, 0:1].view(1, self.num_anchors, 1, 1)
            anchor_h = scaled_anchors[:, 1:2].view(1, self.num_anchors, 1, 1)

            self.register_buffer('grid_x', grid_x)
            self.register_buffer('grid_y', grid_y)
            self.register_buffer('anchor_w', anchor_w)
            self.register_buffer('anchor_h', anchor_h)
        
        pred_boxes = torch.zeros_like(pred[..., :4])
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h) * self.anchor_h
        pred_boxes = pred_boxes * self.stride
        output = torch.cat((pred_boxes.view(batch_size, -1, 4), conf.view(batch_size, -1, 1), class_pred.view(batch_size, -1, self.num_classes)), -1),
        return output
    
class YOLODetector(nn.Module):
    def __init__(self, num_classes, img_dim=416):
        super(YOLODetector, self).__init__()
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.fpn = nn.ModuleList([nn.Conv2d(2048, 512, kernel_size=1), nn.Conv2d(1024, 512, kernel_size=1), nn.Conv2d(512, 256, kernel_size=1), nn.Conv2d(256, 128, kernel_size=1)])
        self.yolo = nn.ModuleList([YOLO(self.anchors[i], self.num_classes, self.img_dim) for i in range(3)])
        self.pred_convs = nn.ModuleList([nn.Conv2d(512, len(self.anchors[0]) * (self.num_classes + 5), kernel_size=1) for _ in range(3)])
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = []
        large_features = self.fpn[0](features)
        large_output = self.pred_convs[0](large_features)
        large_detection = self.yolo[0](large_output)
        outputs.append(large_detection)
        medium_features = F.interpolate(large_features, scale_factor=2, mode='bilinear', align_corners=False)
        medium_features = self.fpn[1](features)
        medium_output = self.pred_convs[1](medium_features)
        medium_detection = self.yolo[1](medium_output)
        outputs.append(medium_detection)
        small_features = F.interpolate(medium_features, scale_factor=2, mode='bilinear', align_corners=False)
        small_features = self.fpn[2](features)
        small_output = self.pred_convs[2](small_features)
        small_detection = self.yolo[2](small_output)
        outputs.append(small_detection)
        return torch.cat(outputs, 1)
    
class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss(reduction='sum')
    
    def forward(self, pred, target):
        n_batch = pred.size(0)
        pred_boxes = pred[..., :4]
        pred_conf = pred[..., 4]
        pred_cls = pred[..., 5:]
        loss_x = 0.0
        loss_y = 0.0 
        loss_w = 0.0
        loss_h = 0.0
        loss_conf = 0.0
        loss_cls = 0.0

        for i in range(n_batch):
            valid = (target['classes'][i] != -1).nonzero(as_tuple=True)[0]
            if len(valid) == 0:
                loss_conf += self.lambda_noobj * self.bce(pred_conf[i], torch.zeros_like(pred_conf[i]))
                continue
            gt_boxes = target['bboxes'][i][valid]
            gt_classes = target['classes'][i][valid]
            for j in range(len(gt_boxes)):
                gt_box = gt_boxes[j]
                gt_class = gt_classes[j]
                gt_x_center = (gt_box[0] + gt_box[2]) / 2
                gt_y_center = (gt_box[1] + gt_box[3]) / 2
                gt_w = gt_box[2] - gt_box[0]
                gt_h = gt_box[3] - gt_box[1]
                gt_yolo = torch.tensor([gt_x_center, gt_y_center, gt_w, gt_h], device=pred.device)
                ious = self.bbox_iou(pred_boxes[i], gt_yolo.unsqueeze(0))
                best_idx = torch.argmax(ious)
                loss_x += self.lambda_coord * self.mse(pred_boxes[i, best_idx, 0], gt_yolo[0])
                loss_y += self.lambda_coord * self.mse(pred_boxes[i, best_idx, 1], gt_yolo[1])
                loss_w += self.lambda_coord * self.mse(pred_boxes[i, best_idx, 2], gt_yolo[2])
                loss_h += self.lambda_coord * self.mse(pred_boxes[i, best_idx, 3], gt_yolo[3])
                loss_conf += self.bce(pred_conf[i, best_idx], torch.ones_like(pred_conf[i, best_idx]))
                target_cls = torch.zeros_like(pred_cls[i, best_idx])
                target_cls[gt_class] = 1.0
                loss_cls += self.bce(pred_cls[i, best_idx], target_cls)

                obj_mask = torch.zeros_like(pred_conf[i])
                obj_mask[best_idx] = 1.0
                noobj_mask = 1.0 - obj_mask
                loss_conf += self.lambda_noobj * self.bce(pred_conf[i] * noobj_mask, torch.zeros_like(pred_conf[i]))
        
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        return loss / n_batch
    
    def bbox_iou(self, b1, b2):
        b1_x1 = b1[:, 0] - b1[:, 2] / 2
        b1_y1 = b1[:, 1] - b1[:, 3] / 2
        b1_x2 = b1[:, 0] + b1[:, 2] / 2
        b1_y2 = b1[:, 1] + b1[:, 3] / 2
        b2_x1 = b2[:, 0] - b2[:, 2] / 2
        b2_y1 = b2[:, 1] - b2[:, 3] / 2
        b2_x2 = b2[:, 0] + b2[:, 2] / 2
        b2_y2 = b2[:, 1] + b2[:, 3] / 2

        inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
        inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
        inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
        inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
        inter_rect_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        area_union = b1_area.unsqueeze(1) + b2_area - inter_rect_area
        iou = inter_rect_area / (area_union + 1e-16)
        return iou.squeeze(1)
    
    def nms(preds, conf_thres=0.5, nms_thres=0.4):
        n_batch = preds.size(0)
        output = [None] * n_batch
        for i in range(n_batch):
            pred = preds[i]
            mask = pred[:, 4] > conf_thres
            pred = pred[mask]
            if not pred.size(0):
                continue
            conf, idx = torch.max(pred[:, 5], dim=1)
            detections = torch.cat((pred[:, :2] - pred[:, 2:4] / 2, pred[:, :2] + pred[:, 2:4] / 2, pred[:, 4].unsqueeze(1), conf.unsqueeze(1), idx.float().unsqueeze(1)), dim=1)
            unique_classes = detections[:, -1].unique()
            output_img = []
            for cls in unique_classes:
                detections_cls = detections[detections[:, -1] == cls]
                _, sort_idx = torch.sort(detections_cls[:, 4], descending=True)
                detections_cls = detections_cls[sort_idx]
                max_detections = []
                while detections_cls.size(0):
                    max_detections.append(detections_cls[0].unsqueeze(0))
                    if len(detections_cls) == 1:
                        break
                    ious = iou_for_nms(max_detections[-1][:, :4], detections_cls[1:, :4])
                    detections_cls = detections_cls[1:][ious < nms_thres]
                if len(max_detections) > 0:
                    output_img.append(torch.cat(max_detections))
            if len(output_img) > 0:
                output[i] = torch.cat(output_img)
        return output
    
    def iou_for_nms(box1, box2):
        b1_x1 = box1[0, 0]
        b1_y1 = box1[0, 1]
        b1_x2 = box1[0, 2]
        b1_y2 = box1[0, 3]
        b2_x1 = box2[:, 0]
        b2_y1 = box2[:, 1]
        b2_x2 = box2[:, 2]
        b2_y2 = box2[:, 3]

        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        
        inter_rect_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        area_union = b1_area + b2_area - inter_rect_area
        iou = inter_rect_area / (area_union + 1e-16)
        return iou
    
    def train(model, train_loader, val_loader, num_epochs=5, lr=0.001, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = loss()
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, data in train_loader:
                i = i.to(device)
                data['classes'] = data['classes'].to(device)
                data['bboxes'] = data['bboxes'].to(device)
                optimizer.zero_grad()
                preds = model(i)
                loss = criterion(preds, data)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * i.size(0)
            e_loss = running_loss / len(train_loader.dataset)
            train_losses.append(e_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {e_loss:.4f}")
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, data in val_loader:
                    i = i.to(device)
                    data['classes'] = data['classes'].to(device)
                    data['bboxes'] = data['bboxes'].to(device)
                    preds = model(i)
                    loss = criterion(preds, data)
                    val_loss += loss.item() * i.size(0)
            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss : {e_loss:.4f}")

            torch.save(model.state_dict(), "yolo_kitti.pth")
            sample_img = []
            model.eval()
            with torch.no_grad():
                for i, data in val_loader:
                    i = i.to(device)
                    preds = model(i)
                    detections = nms(preds)
                    for j in range(len(i)):
                        if detections[j] is not None:
                            img_tensor = i[j].cpu()
                            img_np = img_tensor.permute(1, 2, 0).numpy()
                            img_np = (img_np * 255).astype("uint8")
                            for det in detections[j]:
                                x1, y1, x2, y2 = det[:4].cpu().numpy()
                                conf = det[4].item()
                                cls = det[5].item()
                                id = int(det[6].item())
                                x1 = int(x1 * img_np.shape[1])
                                y1 = int(y1 * img_np.shape[0])
                                x2 = int(x2 * img_np.shape[1])
                                y2 = int(y2 * img_np.shape[0])
                                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
                                label = f"{class_names[id]} {conf:.2f}"
                                cv2.putText(img_np, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            sample_img.append(img_np)
                            if len(sample_img) >= 5:
                                break
                    if len(sample_img) >= 5:
                        break
        return model, train_losses, val_losses, sample_img
    def pred_and_vis(model, img_pths, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), conf_thres=0.5):
        model.to(device)
        model.eval()
        from torchvision import transforms
        transform = transforms.Compose([ transforms.ToTensor(), transforms.Resize((416, 416))])
        class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
        with torch.no_grad():
            for img_pth in img_pths:
                img = cv2.imread(img_pth)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = img.shape[:2]
                input_tensor = transform(img_rgb).unsqueeze(0).to(device)
                preds = model(input_tensor)
                detections = nms(preds, conf_thres=conf_thres)
                res_img = img.copy()
                if detections[0] is not None:
                    for det in detections[0]:
                        x1, y1, x2, y2 = det[:4].cpu().numpy()
                        conf = det[4].item()
                        cls = det[5].item()
                        id = int(det[6].item())
                        x1 = int(x1 * orig_w)
                        y1 = int(y1 * orig_h)
                        x2 = int(x2 * orig_w)
                        y2 = int(y2 * orig_h)
                        cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_names[id]} {conf:.2f}"
                        txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(res_img, (x1, y1 - txt_size[1] - 10), (x1 + txt_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(res_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
                plt.title(f"Detection results for {img_pth.split('/')[-1]}")
                plt.axis('off')
                plt.show()
            
            
                    