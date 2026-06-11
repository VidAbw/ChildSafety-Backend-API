"""
Fine-tune YOLOv8s on a knife-detection dataset.

The resulting `best.pt` is a hazard-specialist model. We keep stock yolov8s.pt
for person detection, so person accuracy is not affected.

Run:
    python -m nanny_cam_guardian.training.train --data path/to/data.yaml
    python -m nanny_cam_guardian.training.train --data path/to/data.yaml --epochs 30 --imgsz 640

After training, copy the weights into the project root:
    cp runs/detect/knife_finetune/weights/best.pt ./hazard_yolo.pt

Then yolo.py will pick it up automatically (it checks for ./hazard_yolo.pt).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_BASE_MODEL = "yolov8s.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
RUN_NAME = "knife_finetune"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on a custom dataset.")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--base", default=DEFAULT_BASE_MODEL, help="Base weights (.pt)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--name", default=RUN_NAME)
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    print(f"[train] base={args.base}  data={data_yaml}")
    print(f"[train] epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}")

    model = YOLO(args.base)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=10,        # early-stop if no improvement for 10 epochs
        plots=True,
        verbose=True,
    )

    best = Path("runs") / "detect" / args.name / "weights" / "best.pt"
    print("\n[train] Training complete.")
    print(f"[train] Best weights: {best.resolve()}")
    print("\nNext step — copy the weights so the detector picks them up:")
    print(f"  cp '{best}' ./hazard_yolo.pt")
    print("Then restart the camera loop.")


if __name__ == "__main__":
    main()
