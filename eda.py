
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Dataset path
dataset_path = "/content/drive/MyDrive/liver images"

# Step 1: Create DataFrame
classes = os.listdir(dataset_path)
image_paths, labels = [], []

for cls in classes:
    cls_path = os.path.join(dataset_path, cls)
    if os.path.isdir(cls_path):
        for img_file in os.listdir(cls_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cls_path, img_file))
                labels.append(cls)

df = pd.DataFrame({'image_path': image_paths, 'label': labels})

# Step 2: Class Distribution Plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='label')
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close()

# Step 3: Image Size Analysis
sizes = []
for path in tqdm(df['image_path'].sample(min(200, len(df)))):
    try:
        img = Image.open(path)
        sizes.append(img.size)
    except:
        continue
sizes_df = pd.DataFrame(sizes, columns=["width", "height"])
sizes_df.describe().to_csv("image_sizes.csv")

# Step 4: Sample Images
def show_images(df, n=9):
    plt.figure(figsize=(10, 10))
    samples = df.sample(n)
    for i, row in enumerate(samples.iterrows()):
        path = row[1]['image_path']
        label = row[1]['label']
        try:
            img = Image.open(path)
            plt.subplot(3, 3, i + 1)
            plt.imshow(img)
            plt.title(label)
            plt.axis('off')
        except:
            continue
    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.close()

show_images(df)

# Save summary
df.to_csv("eda_dataframe.csv", index=False)
print("✅ EDA complete. Files saved: class_distribution.png, image_sizes.csv, sample_images.png, eda_dataframe.csv")
