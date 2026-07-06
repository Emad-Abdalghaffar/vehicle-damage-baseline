import json
import os
import random
from sklearn.model_selection import train_test_split

COCO_ANNOTATIONS = 'annotations/instances_train.json'
OUTPUT_DIR = 'data'
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(COCO_ANNOTATIONS, 'r') as f:
        coco = json.load(f)
    
    image_ids = [img['id'] for img in coco['images']]
    print(f"Total images: {len(image_ids)}")
    
    train_ids, temp_ids = train_test_split(
        image_ids, 
        test_size=(VAL_RATIO + TEST_RATIO), 
        random_state=RANDOM_SEED
    )
    
    val_ids, test_ids = train_test_split(
        temp_ids, 
        test_size=0.5,  # 50% of temp = 15% of total
        random_state=RANDOM_SEED
    )
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # splits as text files with image IDs
    with open(os.path.join(OUTPUT_DIR, 'train.txt'), 'w') as f:
        for img_id in train_ids:
            f.write(f"{img_id}\n")
    with open(os.path.join(OUTPUT_DIR, 'val.txt'), 'w') as f:
        for img_id in val_ids:
            f.write(f"{img_id}\n")
    with open(os.path.join(OUTPUT_DIR, 'test.txt'), 'w') as f:
        for img_id in test_ids:
            f.write(f"{img_id}\n")
    
    # mapping from image ID to filename (useful for conversion)
    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    with open(os.path.join(OUTPUT_DIR, 'image_id_to_filename.json'), 'w') as f:
        json.dump(id_to_filename, f, indent=2)
    
    print(f"Split files saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()