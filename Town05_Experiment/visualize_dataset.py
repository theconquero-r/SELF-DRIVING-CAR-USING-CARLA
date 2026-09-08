import cv2
import os
import random
import pygame

DATASET_DIR = "/home/conqueror/AI_STORAGE/auto_yolo_dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images/train")
LBL_DIR = os.path.join(DATASET_DIR, "labels/train")

images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]

if len(images) == 0:
    print("No images found in dataset!")
    exit(0)

pygame.init()
display = pygame.display.set_mode((640, 640))
pygame.display.set_caption("Dataset Visualizer (Press SPACE for next image)")

classes = ["Pedestrian", "Vehicle", "Traffic Light", "Traffic Sign"]
colors = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255)] # Cyan, Blue, Red, Magenta

def load_random_image():
    random_img = random.choice(images)
    img_path = os.path.join(IMG_DIR, random_img)
    lbl_path = os.path.join(LBL_DIR, random_img.replace('.jpg', '.txt'))

    img = cv2.imread(img_path)
    if img is None: return
    h, w, _ = img.shape

    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                
                # YOLO format to pixel coordinates
                x_min = int((cx - bw / 2) * w)
                y_min = int((cy - bh / 2) * h)
                x_max = int((cx + bw / 2) * w)
                y_max = int((cy + bh / 2) * h)
                
                color = colors[cls_id]
                label = classes[cls_id]
                
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(img, label, (x_min, max(0, y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert OpenCV BGR to Pygame RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    pygame.display.flip()
    print(f"Displaying {random_img}... (Press SPACE for next)")

load_random_image()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE or event.key == pygame.K_RIGHT:
                load_random_image()

pygame.quit()
