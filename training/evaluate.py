import argparse
import os
import sys
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained YOLO Model on Validation/Test Set")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset/data.yaml",
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate ('val' or 'test')",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to evaluate on ('cpu', '0', etc.)",
    )
    return parser.parse_args()


def evaluate():
    args = parse_args()
    model_path = args.model
    if not os.path.exists(model_path):
        # Fallback check
        if os.path.exists("models/yolo11n.pt"):
            model_path = "models/yolo11n.pt"
        else:
            model_path = "yolo11n.pt"
        print(f"[Notice] Model checkpoint not found at {args.model}. Evaluating baseline {model_path}...")

    # Resolve dataset path
    data_path = args.data
    if os.path.exists(args.data):
        data_path = os.path.abspath(args.data)
    elif os.path.exists(os.path.join("data", "dataset", args.data)):
        data_path = os.path.abspath(os.path.join("data", "dataset", args.data))

    print("=" * 60)
    print(" YOLO Model Validation & Evaluation")
    print("=" * 60)
    print(f" Model Checkpoint : {model_path}")
    print(f" Dataset YAML     : {data_path}")
    print(f" Evaluation Split : {args.split}")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        save_json=True,
    )

    print("\n" + "=" * 60)
    print(" Evaluation Summary Metrics:")
    print("-" * 60)
    print(f" Precision (P) : {metrics.box.mp:.4f}")
    print(f" Recall (R)    : {metrics.box.mr:.4f}")
    print(f" mAP@50        : {metrics.box.map50:.4f}")
    print(f" mAP@50-95     : {metrics.box.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
