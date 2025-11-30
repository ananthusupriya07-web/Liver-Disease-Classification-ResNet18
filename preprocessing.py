
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
from collections import Counter
from torchvision.datasets.folder import pil_loader

# Constants
DATASET_PATH = "/content/drive/MyDrive/liver images"
IMG_SIZE = (224, 224)
SEED = 42

# Step 1: Collect paths and labels
image_paths, labels = [], []
classes = sorted(os.listdir(DATASET_PATH))

for cls in classes:
    cls_path = os.path.join(DATASET_PATH, cls)
    if os.path.isdir(cls_path):
        for img_file in os.listdir(cls_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cls_path, img_file))
                labels.append(cls)

df = pd.DataFrame({'image_path': image_paths, 'label': labels})

# Step 2: 70:15:15 Stratified Split
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=SEED)

print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))

# Step 3: Define Transforms
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

base_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Step 4: Preprocess & Save to .npz
def preprocess_and_save(df, split_name, transform):
    data = []
    labels = []
    label_to_idx = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        try:
            img = pil_loader(row['image_path'])
            img_tensor = transform(img)
            data.append(img_tensor.numpy())
            labels.append(label_to_idx[row['label']])
        except Exception as e:
            print(f"Skipped {row['image_path']} due to error: {e}")
            continue

    data = np.array(data)
    labels = np.array(labels)
    np.savez(f"{split_name}_preprocessed.npz", data=data, labels=labels)
    print(f"✅ {split_name} set saved with shape: {data.shape}, class counts: {Counter(labels)}")

# Run preprocessing
preprocess_and_save(train_df, "train", train_transform)
preprocess_and_save(val_df, "val", base_transform)
preprocess_and_save(test_df, "test", base_transform)
