from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Run inference on a single image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='runs/detect/cardd_yolo11m/weights/best.pt', help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save_dir', type=str, default='outputs/inference', help='Output directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return
    
    model = YOLO(args.model)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # inference
    results = model(
        source=args.image,
        conf=args.conf,
        save=True,
        project=args.save_dir,
        name='run',
        exist_ok=True
    )
    
    # results
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            print(f"\nDetected {len(boxes)} instances:")
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                print(f"  {i+1}: Class {cls} (confidence: {conf:.3f}) at {xyxy}")
    
    print(f"\nResults saved to {args.save_dir}/run/")

if __name__ == '__main__':
    main()