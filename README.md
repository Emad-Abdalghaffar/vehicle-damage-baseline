## vehicle damage detection baseline
A narrow proof-of-concept for car damage detection using YOLOv11 on the CarDD dataset.

## installation
git clone https://github.com/Emad-Abdalghaffar/vehicle-damage-baseline.git
cd vehicle-damage-baseline
pip install -r requirements.txt

## dataset download
Download CarDD from https://cardd-ustc.github.io.
Place the extracted folder with images/ and annotations/ inside the project root.
Expected structure:

	vehicle-damage-baseline/
	├── images/                    (CarDD original images)
	└── annotations/
		└── instances_train.json   (CarDD COCO annotations)
		
## repository sstructure

	vehicle-damage-baseline/
	├── README.md
	├── requirements.txt
	├── config/
	│   └── cardd.yaml
	├── scripts/
	│   ├── split_dataset.py
	│   ├── convert_coco_to_yolo.py
	│   └── visualize_labels.py
	├── src/
	│   ├── train.py
	│   ├── evaluate.py
	│   └── inference.py
	│
	├── images/
	├── annotations/
	│
	└── data/                     (auto-created by scripts)
		├── images/               (train/val/test subfolders)
		└── labels/               (train/val/test subfolders)
		
		
## data preparation
	Run these scripts in order:
	# 1. split dataset into train/val/test (70/15/15)
	python scripts/split_dataset.py
	# 2. convert COCO annotations to YOLO format
	python scripts/convert_coco_to_yolo.py
	# 3. sanity check – visualize labels
	python scripts/visualize_labels.py --split train
	
	
## issue/start a training session
	python src/train.py
	
## trained weights evaluation
	python src/evaluate.py
	
## inference
	python src/inference.py --image [path/to/image.jpg] --model runs/detect/cardd_yolo11m/weights/best.pt
	
	
	
	
## what I would do next with more time
	> implement Active Learning to simulate labeling efficiency with the 200-vehicle budget.
	> replace YOLO backbone with DINOv2 (frozen) for better robustness to lighting/angle/depth variation.
	> integrate Ordinal Regression for the Effort Grade.
	> build an Uncertainty Wrapper (e.g.: Monte Carlo Dropout) to generate confidence intervals for each prediction.
