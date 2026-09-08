import os
import shutil
import random
import glob

def split_dataset():
    images = sorted(glob.glob("dataset/images/*.png"))
    
    # Create YOLO directory structure
    dirs = [
        "yolo_dataset/train/images",
        "yolo_dataset/train/labels",
        "yolo_dataset/val/images",
        "yolo_dataset/val/labels"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    # Shuffle for random train/val split
    random.seed(42)
    random.shuffle(images)
    
    # 80% train, 20% val
    split_index = int(len(images) * 0.8)
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    def copy_files(image_paths, split_type):
        for img_path in image_paths:
            base_name = os.path.basename(img_path)
            label_name = base_name.replace('.png', '.txt')
            
            label_path = f"dataset/labels/{label_name}"
            
            # Only copy if both image and label exist
            if os.path.exists(label_path):
                shutil.copy(img_path, f"yolo_dataset/{split_type}/images/{base_name}")
                shutil.copy(label_path, f"yolo_dataset/{split_type}/labels/{label_name}")

    print("Splitting into 80% Train and 20% Val...")
    copy_files(train_images, "train")
    copy_files(val_images, "val")
    print(f"Data split complete! Train: {len(train_images)}, Val: {len(val_images)}")

if __name__ == "__main__":
    split_dataset()
