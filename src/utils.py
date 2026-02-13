import os
import shutil
import random

# Source dataset path
SOURCE_DIR = "../dataset/PlantVillage"

# Destination paths
TRAIN_DIR = "../dataset/train"
TEST_DIR = "../dataset/test"

# Classes we selected
CLASSES = {
    "Tomato_Early_blight": "Early_blight",
    "Tomato_Late_blight": "Late_blight",
    "Tomato_Septoria_leaf_spot": "Septoria_leaf_spot",
    "Tomato_healthy": "healthy"
}

SPLIT_RATIO = 0.8  # 80% train, 20% test

for original_class, new_class in CLASSES.items():
    source_path = os.path.join(SOURCE_DIR, original_class)
    images = os.listdir(source_path)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_index]
    test_images = images[split_index:]

    # Copy training images
    for img in train_images:
        shutil.copy(
            os.path.join(source_path, img),
            os.path.join(TRAIN_DIR, new_class, img)
        )

    # Copy testing images
    for img in test_images:
        shutil.copy(
            os.path.join(source_path, img),
            os.path.join(TEST_DIR, new_class, img)
        )

    print(f"{new_class} done ✅")

print("Dataset splitting completed 🚀")
