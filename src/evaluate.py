from ultralytics import YOLO
import json
import os

def main():
    # Load the best model from training
    model_path = 'runs/detect/cardd_yolo11m/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run training first.")
        return
    
    model = YOLO(model_path)
    
    # Evaluate on the held-out test set
    results = model.val(
        data='config/cardd.yaml',
        split='test',
        imgsz=640,
        batch=16,
        conf=0.001, # Include all predictions for mAP calculation
        iou=0.5,
        save_json=True,
        plots=True,
        project='runs/detect/cardd_yolo11m',
        name='evaluate_test'
    )
    
    # metrics
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
        'f1': results.box.f1,
        'total_images': results.nt,
        'per_class_ap': results.box.ap_class_index.tolist() if hasattr(results.box, 'ap_class_index') else None,
    }
    
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n=== Test Set Evaluation Results ===")
    print(f"mAP@0.50: {metrics['mAP50']:.4f}")
    print(f"mAP@0.50:0.95: {metrics['mAP50-95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Total images evaluated: {metrics['total_images']}")
    print("\nMetrics saved to outputs/metrics.json")

if __name__ == '__main__':
    main()