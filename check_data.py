import os

for split in ["train", "val", "test"]:
    path = f"organized_data/{split}"
    total_images = 0
    classes = 0
    
    for class_folder in os.listdir(path):
        class_path = os.path.join(path, class_folder)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            total_images += count
            classes += 1
            if classes <= 3:  # Show first 3
                print(f"{split}/{class_folder}: {count} images")
    
    print(f"\n{split.upper()}: {classes} classes, {total_images} total images\n")
