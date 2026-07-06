import json
import os
import shutil
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm

COCO_ANNOTATIONS = 'annotations/instances_train.json'
SPLIT_DIR = 'data'
ORIGINAL_IMAGES_DIR = 'images'
OUTPUT_IMAGES_DIR = 'data/images'
OUTPUT_LABELS_DIR = 'data/labels'

def convert_coco_to_yolo(split_name, image_ids, coco, id_to_filename):
    """Convert a split of COCO annotations to YOLO format."""
    images_dir = os.path.join(OUTPUT_IMAGES_DIR, split_name)
    labels_dir = os.path.join(OUTPUT_LABELS_DIR, split_name)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for img_id in tqdm(image_ids, desc=f"Converting {split_name}"):
        img_info = coco.loadImgs([img_id])[0]
        filename = id_to_filename.get(str(img_id), img_info['file_name'])
        img_width = img_info['width']
        img_height = img_info['height']
        
        src_img_path = os.path.join(ORIGINAL_IMAGES_DIR, filename)
        dst_img_path = os.path.join(images_dir, filename)
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image not found: {src_img_path}")
            continue
        
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        annotations = coco.loadAnns(ann_ids)
        
        # YOLO labels
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                cat_id = ann['category_id'] - 1  # COCO IDs are 1-indexed
                
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # OLO format: normalized center x, center y, width, height
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width_norm = w / img_width
                height_norm = h / img_height
                
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width_norm = max(0, min(1, width_norm))
                height_norm = max(0, min(1, height_norm))
                
                f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

def main():
    coco = COCO(COCO_ANNOTATIONS)
    
    with open(os.path.join(SPLIT_DIR, 'train.txt'), 'r') as f:
        train_ids = [int(line.strip()) for line in f.readlines()]
    with open(os.path.join(SPLIT_DIR, 'val.txt'), 'r') as f:
        val_ids = [int(line.strip()) for line in f.readlines()]
    with open(os.path.join(SPLIT_DIR, 'test.txt'), 'r') as f:
        test_ids = [int(line.strip()) for line in f.readlines()]
    
    with open(os.path.join(SPLIT_DIR, 'image_id_to_filename.json'), 'r') as f:
        id_to_filename = json.load(f)
    
    convert_coco_to_yolo('train', train_ids, coco, id_to_filename)
    convert_coco_to_yolo('val', val_ids, coco, id_to_filename)
    convert_coco_to_yolo('test', test_ids, coco, id_to_filename)
    
    print("Conversion complete! Dataset ready at ./data/")

if __name__ == '__main__':
    main()