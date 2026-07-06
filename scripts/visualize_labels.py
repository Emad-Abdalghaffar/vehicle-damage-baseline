import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def draw_yolo_labels(image_path, label_path, class_names):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    if not os.path.exists(label_path):
        print(f"Label not found: {label_path}")
        return cv2.imread(image_path)
    
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        box_w = float(parts[3]) * w
        box_h = float(parts[4]) * h
        
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)
        
        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--num_samples', type=int, default=5, help='Number of images to visualize')
    args = parser.parse_args()
    
    class_names = ['dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat']
    
    images_dir = os.path.join('data', 'images', args.split)
    labels_dir = os.path.join('data', 'labels', args.split)
    
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} not found. Run conversion first.")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files = image_files[:args.num_samples]
    
    output_dir = 'outputs/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        vis_img = draw_yolo_labels(img_path, label_path, class_names)
        if vis_img is not None:
            out_path = os.path.join(output_dir, f'{args.split}_{img_file}')
            cv2.imwrite(out_path, vis_img)
            print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()