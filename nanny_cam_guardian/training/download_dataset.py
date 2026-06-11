"""
Download a knife-detection dataset from Roboflow Universe in YOLOv8 format.

─────────────────────────────────────────────────────────────────────────────
HOW TO FIND YOUR DATASET SLUGS (do this once):
  1. Go to https://universe.roboflow.com and search "knife detection"
  2. Open any dataset (pick one with 1000+ images and "YOLOv8" format)
  3. Click "Download Dataset" → select format "YOLOv8" → "Get Download Code"
  4. Roboflow shows a snippet like:
       rf.workspace("my-workspace").project("knife-det").version(3)
  5. Copy those three slugs and either:
       a) Pass as args:  --workspace my-workspace --project knife-det --version 3
       b) Add to .env:   RF_WORKSPACE=my-workspace
                         RF_PROJECT=knife-det
                         RF_VERSION=3
─────────────────────────────────────────────────────────────────────────────

Setup (one-time):
    1. Sign up free at https://app.roboflow.com (Google sign-in works)
    2. Settings → Roboflow API → copy your Private API key
    3. Add to .env:    ROBOFLOW_API_KEY=your_key_here
    4. pip install roboflow==1.1.50

Run:
    python -m nanny_cam_guardian.training.download_dataset --workspace <ws> --project <proj> --version <ver>
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

FORMAT  = "yolov8"
OUT_DIR = Path(__file__).resolve().parent / "datasets"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=os.getenv("RF_WORKSPACE"),
                        help="Roboflow workspace slug (from the dataset URL)")
    parser.add_argument("--project",   default=os.getenv("RF_PROJECT"),
                        help="Roboflow project slug")
    parser.add_argument("--version",   default=os.getenv("RF_VERSION", "1"), type=int,
                        help="Dataset version number")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY missing.\n"
            "  Get it at https://app.roboflow.com → Settings → Roboflow API\n"
            "  Then add it to your .env file: ROBOFLOW_API_KEY=your_key_here"
        )

    if not args.workspace or not args.project:
        raise SystemExit(
            "\nMissing --workspace / --project.\n"
            "\nHow to find them:\n"
            "  1. Go to https://universe.roboflow.com and search 'knife detection'\n"
            "  2. Open a dataset and click Download Dataset → YOLOv8 → Get Download Code\n"
            "  3. Copy the slugs from the snippet Roboflow shows, then run:\n\n"
            "     python -m nanny_cam_guardian.training.download_dataset \\\n"
            "         --workspace YOUR_WORKSPACE \\\n"
            "         --project YOUR_PROJECT \\\n"
            "         --version 1\n"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    out_path = OUT_DIR / f"{args.project}-v{args.version}"
    print(f"[download] Pulling {args.workspace}/{args.project} v{args.version} → {out_path}")
    dataset = version.download(FORMAT, location=str(out_path))
    print(f"[download] Done. Dataset at: {dataset.location}")
    print("\nNext step:")
    print(f"  python -m nanny_cam_guardian.training.train --data \"{dataset.location}/data.yaml\"")


if __name__ == "__main__":
    main()
