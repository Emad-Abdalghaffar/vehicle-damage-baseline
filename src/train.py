from ultralytics import YOLO
import torch
import os

def main():
    model = YOLO('yolo11m.pt')
    
    os.makedirs('runs/detect', exist_ok=True)
    
    # Train the model
    results = model.train(
        data='config/cardd.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else 'cpu',
        
        # Optimization
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        
        # Regularization
        dropout=0.1,
        weight_decay=0.0005,
        
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Training control
        patience=25,
        cos_lr=True,
        amp=True,
        multi_scale=True,
        val=True,
        project='runs/detect',
        name='cardd_yolo11m',
        exist_ok=True
    )
    
    print(f"Training complete. Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()