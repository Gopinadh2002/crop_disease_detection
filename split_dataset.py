import os
import shutil
import random
from pathlib import Path

# Paths
data_path = "data"
output_path = "organized_data"
train_split = 0.8
val_split = 0.1
test_split = 0.1

print("Starting dataset split...")

# Step 1: Create output directory structure
for split in ["train", "val", "test"]:
    os.makedirs(output_path + "/" + split, exist_ok=True)
    print(f"Created {output_path}/{split}/")

# Step 2: Process each disease class folder
for disease_folder in sorted(os.listdir(data_path)):
    disease_path = os.path.join(data_path, disease_folder)
    
    # Skip if not a directory
    if not os.path.isdir(disease_path):
        continue
    
    print(f"\nProcessing: {disease_folder}")
    
    # Create disease subfolder in train/val/test
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, split, disease_folder), exist_ok=True)
    
    # Get all image files
    images = os.listdir(disease_path)
    total_images = len(images)
    
    # Shuffle randomly
    random.shuffle(images)
    
    # Calculate split indices
    train_count = int(total_images * train_split)
    val_count = int(total_images * val_split)
    
    # Split into train, val, test
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]
    
    # Copy train images
    for img in train_images:
        src = os.path.join(disease_path, img)
        dst = os.path.join(output_path, "train", disease_folder, img)
        shutil.copy(src, dst)
    
    # Copy val images
    for img in val_images:
        src = os.path.join(disease_path, img)
        dst = os.path.join(output_path, "val", disease_folder, img)
        shutil.copy(src, dst)
    
    # Copy test images
    for img in test_images:
        src = os.path.join(disease_path, img)
        dst = os.path.join(output_path, "test", disease_folder, img)
        shutil.copy(src, dst)
    
    # Print summary
    print(f"  Total: {total_images} images")
    print(f"  Train: {len(train_images)} ({train_split*100:.0f}%)")
    print(f"  Val: {len(val_images)} ({val_split*100:.0f}%)")
    print(f"  Test: {len(test_images)} ({test_split*100:.0f}%)")

print("\n✅ Dataset split complete!")
