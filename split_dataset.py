import os
import shutil
import random

random.seed(42)

SOURCE_DIR = "raw_dataset"
DEST_DIR = "data"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    train_split = int(len(images) * TRAIN_RATIO)
    val_split = int(len(images) * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": images[:train_split],
        "val": images[train_split:val_split],
        "test": images[val_split:]
    }

    for split_name, split_images in splits.items():
        split_class_dir = os.path.join(DEST_DIR, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copyfile(src, dst)

print("Dataset split completed successfully.")