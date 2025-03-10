import torch
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

COLORS = [
    (0, 0, 255),
    (0, 128, 255), 
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
    (128, 128, 128),
    (0, 0, 0)
]

def bbox(image, out, threshold=0.5, orig_size=None):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy() * 255
        image = image.astype(np.uint8)

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]

    if orig_size:
        orig_w, orig_h = orig_size
        x = orig_w / w
        y = orig_h / h
    else:
        x, y = 1.0, 1.0

    scores = out['classes']
    bbox = out['bbox']

    if isinstance(scores, torch.Tensor):
        pred_class = torch.argmax(scores).item()
        confidence = torch.softmax(scores, dim=0)[pred_class].item()
    else:
        pred_class = np.argmax(scores)
        confidence = float(np.exp(scores[pred_class]) / np.sum(np.exp(scores)))
    
    # Only draw if confidence is above threshold
    if confidence >= threshold:
        # Get bounding box coordinates (normalized to [0,1])
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(bbox[0] * w * x)
        y1 = int(bbox[1] * h * y)
        x2 = int(bbox[2] * w * x)
        y2 = int(bbox[3] * h * y)
        
        # Draw bounding box
        color = COLORS[pred_class]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        
        # Add label with class name and confidence
        label = f"{CLASSES[pred_class]}: {confidence:.2f}"
        cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convert back to RGB for display
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def visualize_batch(images, out, targets=None, threshold=0.5, max=4):
    batch_size = min(images.shape[0], max)
    fig, axes = plt.subplots(batch_size, 1, figsize=(10, 5 * batch_size))
    
    # Handle case where batch_size is 1
    if batch_size == 1:
        axes = [axes]
    
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img_with_boxes = img.copy()
        h, w = img.shape[:2]
        class_scores = out['classes'][i]
        bbox = out['bboxes'][i]
        pred_class = torch.argmax(class_scores).item()
        confidence = torch.softmax(class_scores, dim=0)[pred_class].item()
        
        if confidence >= threshold:
            x1 = int(bbox[0].item() * w)
            y1 = int(bbox[1].item() * h)
            x2 = int(bbox[2].item() * w)
            y2 = int(bbox[3].item() * h)
            
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{CLASSES[pred_class]}: {confidence:.2f}"
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if targets is not None:
            gt_classes = targets['classes'][i]
            gt_boxes = targets['bboxes'][i]
            
            for j in range(len(gt_classes)):
                if gt_classes[j].item() == -1:
                    continue  
                
                gt_class = gt_classes[j].item()
                gt_box = gt_boxes[j]
                
                x1 = int(gt_box[0].item() * w)
                y1 = int(gt_box[1].item() * h)
                x2 = int(gt_box[2].item() * w)
                y2 = int(gt_box[3].item() * h)
                
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"GT: {CLASSES[gt_class]}"
                cv2.putText(img_with_boxes, label, (x1, y1 - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        axes[i].imshow(img_with_boxes)
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def eval(model, val_loader, num_samples=5, threshold=0.5, device=torch.device('cpu')):
    model.eval()
    samples_processed = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            # Move to device
            images = images.to(device)
            
            # Get model predictions
            outputs = model(images)
            
            # Visualize the batch
            visualize_batch(images, outputs, targets, threshold)
            
            samples_processed += images.shape[0]
            if samples_processed >= num_samples:
                break

def save_detection_image(image_path, model, output_path=None, threshold=0.5, device=torch.device('cpu')):
    # Set model to evaluation mode
    model.eval()
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # Apply the same transforms as in the dataset
    transform = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
    ])
    
    # Transform the image
    img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Get the first (and only) item from the batch
    output = {
        'classes': outputs['classes'][0],
        'bboxes': outputs['bboxes'][0]
    }
    
    # Convert tensor back to numpy for visualization
    img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Visualize detection
    result_img = bbox(img_np, output, threshold, original_size)
    
    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_detection{ext}"
    
    # Save the result
    plt.imsave(output_path, result_img)
    
    return output_path 