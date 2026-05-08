import os

data_path = "data"

# Count images per class
classes = {}
for class_folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, class_folder)
    if os.path.isdir(folder_path):
        image_count = len(os.listdir(folder_path))
        classes[class_folder] = image_count

print("Classes and image counts:")
for class_name, count in sorted(classes.items()):
    print(f"  {class_name}: {count} images")

print(f"\nTotal classes: {len(classes)}")
print(f"Total images: {sum(classes.values())}")
