import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import utils
from utils import KITTIDataset, kitti_collate_fn
from Network import KITTIObjectDetector, DetectionLoss, train_model
from torch.utils.data import random_split, Subset
import torchvision.transforms as T
# Select a random subset of the dataset

def main(Network, subset):
    #config variables
    experiment1 = False #validation gaussian only
    experiment2 = False #training gaussian only
    experiment3 = False #training and validation gaussian
    experiment4 = False #some noisy some not
    experiment5 = False #day/night domain shift
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define KITTI dataset URL
    KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    ANNOTATION_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"

    # Define dataset directory
    DATASET_DIR = "./kitti_dataset"

    # Download KITTI images and labels
    os.makedirs(DATASET_DIR, exist_ok=True)
    image_zip = os.path.join(DATASET_DIR, "kitti_images.zip")
    label_zip = os.path.join(DATASET_DIR, "kitti_labels.zip")

    utils.download_kitti(KITTI_URL, image_zip)
    utils.download_kitti(ANNOTATION_URL, label_zip)

    # Extract dataset, commented out because only need to do it once
    # utils.extract_kitti(image_zip, DATASET_DIR)
    # utils.extract_kitti(label_zip, DATASET_DIR)

    image_dir = os.path.join(DATASET_DIR, "training", "image_2")
    label_dir = os.path.join(DATASET_DIR, "training", "label_2")


    full_dataset = KITTIDataset(image_dir, label_dir,transform=T.ToTensor())

    print(f"Total dataset size: {len(full_dataset)}")
    #Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    #make a subset of the data if subset = true
    if(subset):
        #Create a subset of full dataset
        subset_size = 500
        print(f"Total dataset size: {len(subset_size)}")
        indices = list(range(subset_size))
        small_dataset = Subset(full_dataset, indices)
        train_dataset, val_dataset = random_split(small_dataset, [400, 100])

    #apply transformations if any of the experiments are being run
    if(experiment1):
        val_dataset = utils.NoisyDataset(val_dataset, noise_mean = 0.0, noise_std = 0.05)
    if(experiment2):
        train_dataset = utils.NoisyDataset(train_dataset, noise_mean = 0.0, noise_std = 0.05)
    if(experiment3):
        val_dataset = utils.NoisyDataset(val_dataset, noise_mean = 0.0, noise_std = 0.05)
        train_dataset = utils.NoisyDataset(train_dataset, noise_mean = 0.0, noise_std =0.05)
    if(experiment4):
        val_dataset = utils.split_half_noisy(val_dataset, noise_mean = 0.0, noise_std = 0.05)
        train_dataset = utils.split_half_noisy(train_dataset, noise_mean = 0.0, noise_std = 0.05)
    if(experiment5):
        val_dataset = utils.NightDataset(val_dataset)




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


    if(Network == "ResNet18"):
        model = KITTIObjectDetector(num_classes=9)
        model2, train_det_accs2, train_cls_accs2, train_full_det_accs2, val_det_accs2, val_cls_accs2, val_full_det_accs2 = (
            train_model(model, train_loader, val_loader, num_epochs=10, lr=0.002, save_path="nighttime.pth", weight_decay=3e-3))

        #Load the trained model that you want
        model = KITTIObjectDetector(num_classes=9)
        model.load_state_dict(torch.load("BestBBox40epochs.pth", map_location=device, weights_only=True))  # Update with correct checkpoint path
        model.to(device)
        model.eval()

        # List of test images
        test_images = [f"./kitti_dataset/training/image_2/{i:06d}.png" for i in range(2000,2010)] #visualize whatever images you want


        # Run inference and visualize results
        utils.predict_and_visualize2(model, test_images)

if __name__ == "__main__":
    main(Network = "ResNet18", subset = False)
