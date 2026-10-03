import argparse
import os
import shutil
import sys
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Model Fine-Tuning & Training Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset/data.yaml",
        help="Path to dataset data.yaml file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained checkpoint to fine-tune (e.g. yolo11n.pt, yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 for AutoBatch)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on ('cpu', '0', etc.)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory for training artifacts",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment name",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a rapid 3-epoch smoke test to verify dataset paths and GPU pipeline",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    
    # Resolve dataset path: check exact path, then data/dataset/ directory, then Ultralytics builtin
    if os.path.exists(args.data):
        data_yaml_path = os.path.abspath(args.data)
    elif os.path.exists(os.path.join("data", "dataset", args.data)):
        data_yaml_path = os.path.abspath(os.path.join("data", "dataset", args.data))
    elif args.data in ("VOC.yaml", "coco8.yaml", "coco.yaml"):
        data_yaml_path = args.data
    else:
        print(f"[Error] Dataset configuration file not found at: {args.data}")
        print("Please check data/dataset/ or supply a valid YAML file (e.g. data/dataset/VOC.yaml)")
        sys.exit(1)

    epochs = 3 if args.smoke_test else args.epochs
    print("=" * 60)
    print(" YOLO Fine-Tuning Training Pipeline")
    print("=" * 60)
    print(f" Dataset YAML : {data_yaml_path}")
    print(f" Base Weights : {args.model}")
    print(f" Epochs       : {epochs} {'(SMOKE TEST MODE)' if args.smoke_test else ''}")
    print(f" Image Size   : {args.imgsz}")
    print(f" Batch Size   : {args.batch}")
    print(f" Project Dir  : {args.project}/{args.name}")
    print("=" * 60)

    # Initialize model with pretrained weights (Transfer Learning)
    model = YOLO(args.model)

    # Start training
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        save=True,
        verbose=True,
    )

    # Post-training: copy best.pt to models/best.pt for easy local deployment
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        if os.path.exists(best_pt):
            dest_dir = "models"
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, "best.pt")
            shutil.copy2(best_pt, dest_path)
            print("\n" + "=" * 60)
            print(f" [SUCCESS] Training complete! Best weights saved to: {dest_path}")
            print(f" Ready for deployment with OpenCV app: python app/main.py --model models/best.pt")
            print("=" * 60)


if __name__ == "__main__":
    train()
