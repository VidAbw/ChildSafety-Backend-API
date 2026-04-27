# Training — Knife Detection Fine-Tune

Improves hazard detection at distance (corner-mounted camera). Stock `yolov8s.pt` only spots a knife when it's huge in the frame; a fine-tuned model handles small/distant blades.

## One-time setup

```bash
pip install roboflow==1.1.50
```

1. Free Roboflow account: https://app.roboflow.com (Google sign-in works)
2. Settings → API → copy **Private API key**
3. Add to `.env`:
   ```
   ROBOFLOW_API_KEY=your_key_here
   ```

## Step 1 — Download dataset

```bash
python -m nanny_cam_guardian.training.download_dataset
```

Downloads ~1–3GB into `nanny_cam_guardian/training/datasets/`. The script prints the path to `data.yaml`.

## Step 2 — Fine-tune

```bash
python -m nanny_cam_guardian.training.train --data nanny_cam_guardian/training/datasets/knife-detection-luuoc-v2/data.yaml
```

Defaults: 50 epochs, 640px, batch 16, early-stop after 10 epochs with no improvement.

**Time estimates:**
- GPU (RTX 3060+): ~30 min
- CPU only: 6–12 hours — strongly recommend a GPU. Use Google Colab free tier if needed (upload the dataset folder, run the same script).

## Step 3 — Activate

```bash
cp runs/detect/knife_finetune/weights/best.pt ./hazard_yolo.pt
```

Restart the camera loop:

```bash
python -m nanny_cam_guardian.detector.capture
```

`yolo.py` auto-detects `./hazard_yolo.pt` and uses it for hazards while keeping stock `yolov8s.pt` for person detection.

## Verify

Hold a knife at 2–3m from the camera. Stock yolov8s usually misses it; the fine-tuned model should draw a red box reliably.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ROBOFLOW_API_KEY missing` | Set it in `.env` |
| OOM during training | Lower `--batch 8` or `--imgsz 416` |
| Person detection got worse | You overwrote yolov8s.pt — restore it; the fine-tuned model should be `hazard_yolo.pt`, separate file |
| Model still misses knives at distance | Fine-tune more epochs, or pick a dataset with more far-away samples on Roboflow Universe |
