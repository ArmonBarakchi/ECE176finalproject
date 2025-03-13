import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as T
from utils import KITTIDataset, kitti_collate_fn, download_kitti, extract_kitti
from yolo import YOLODetector, train, pred_and_vis
import matplotlib.pyplot as plt

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define KITTI dataset URL
    KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    ANNOTATION_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"

    # Define dataset directory
    DATASET_DIR = "./kitti_dataset"

    # Download KITTI images and labels if not already downloaded
    os.makedirs(DATASET_DIR, exist_ok=True)
    image_zip = os.path.join(DATASET_DIR, "kitti_images.zip")
    label_zip = os.path.join(DATASET_DIR, "kitti_labels.zip")

    # Check if dataset files exist, download if needed
    if not os.path.exists(os.path.join(DATASET_DIR, "training", "image_2")):
        download_kitti(KITTI_URL, image_zip)
        download_kitti(ANNOTATION_URL, label_zip)
        
        # Extract dataset
        extract_kitti(image_zip, DATASET_DIR)
        extract_kitti(label_zip, DATASET_DIR)

    # Define image and label directories
    image_dir = os.path.join(DATASET_DIR, "training", "image_2")
    label_dir = os.path.join(DATASET_DIR, "training", "label_2")

    # Create dataset with transformations
    transform = T.Compose([
        T.ToTensor(),
        # Resize to YOLO input size (416x416)
        # We'll handle this in the model to maintain aspect ratio
    ])
    
    full_dataset = KITTIDataset(image_dir, label_dir, transform=transform)
    print(f"Total dataset size: {len(full_dataset)}")

    # Create a subset of the full dataset for faster training
    subset_size = 500  # Use a small subset for demonstration
    indices = list(range(subset_size))
    small_dataset = Subset(full_dataset, indices)
    
    # Split the small dataset into training and validation parts
    train_size = int(0.8 * len(small_dataset))
    val_size = len(small_dataset) - train_size
    train_dataset, val_dataset = random_split(small_dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Create DataLoaders for both training and validation sets
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=kitti_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=kitti_collate_fn
    )

    # Initialize YOLO model
    model = YOLODetector(num_classes=9, img_size=416)
    
    # Check if a pre-trained model exists
    model_path = "yolo_kitti.pth"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training YOLO model...")
        # Train the model
        model, train_losses, val_losses, sample_images = train(
            model, 
            train_loader, 
            val_loader, 
            num_epochs=5,  # Reduced for demonstration
            lr=0.001, 
            device=device
        )
        
        # Plot training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('YOLO Training and Validation Losses')
        plt.legend()
        plt.savefig('yolo_training_loss.png')
        plt.show()
        
        # Display sample predictions
        if sample_images:
            print(f"Displaying {len(sample_images)} sample predictions")
            for i, img in enumerate(sample_images):
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.title(f"Sample Prediction {i+1}")
                plt.axis('off')
                plt.show()

    # Run inference on test images
    print("Running inference on test images...")
    test_images = [
        "./kitti_dataset/training/image_2/000002.png",
        "./kitti_dataset/training/image_2/000001.png",
    ]
    
    # Add more test images if available
    if os.path.exists("./kitti_dataset/training/image_2/000007.png"):
        test_images.append("./kitti_dataset/training/image_2/000007.png")
    if os.path.exists("./kitti_dataset/training/image_2/000016.png"):
        test_images.append("./kitti_dataset/training/image_2/000016.png")
    if os.path.exists("./kitti_dataset/training/image_2/000025.png"):
        test_images.append("./kitti_dataset/training/image_2/000025.png")
    
    # Visualize predictions
    pred_and_vis(model, test_images, device=device, conf_threshold=0.3)

if __name__ == "__main__":
    main()