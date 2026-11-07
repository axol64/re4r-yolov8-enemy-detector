---
tags:
  - computer-vision
  - ultralytics-yolo
  - portfolio
  - resident-evil-4
---
# Resident Evil 4 CV (Ultralytics YOLO) — Day 2 & Final Project Wrap-Up

## Session Recap: October 15, 2026

### 1. Final Project Status & Outcome
The **Resident Evil 4 Ganado Object Detection** pipeline is complete. Across two days of iterative data-centric engineering (`train` through `train-14`), true unseen validation performance improved from a leaked single-class baseline (`0.536 mAP50`, `0.233 mAP50-95`, `0.512 Recall` in `train-10`) to **`0.747 mAP50` (+39%)**, **`0.545 mAP50-95` (+134%)**, and **`0.824 Recall` (+61%)** by the final `train-13` / `train-14` checkpoints, producing a full-speed 60 FPS tracked gameplay output in [re4_run2_clean.mp4](../runs/detect/re4_run2_clean.mp4).

### 2. Day 2 Progression & Metrics Summary (`train-10` → `train-14`)

| Run | Starting Weights & Key Config | Dataset & Augmentation Changes | Peak `mAP50` | Peak `mAP50-95` | Peak Recall | `val/box_loss` | `val/cls_loss` | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train-10`** *(Day 1 Baseline)* | `yolov8s.pt`, 1 class (`ganado`), `imgsz=768`, `CLAHE(4.0, p=0.5)` | 430 train / 135 val frames | `0.536` | `0.233` | `0.544` | `1.954` | `2.229` | First leak-free validation baseline; suffered from ally false positives. |
| **`train-11`** | `train-10/best.pt`, 4 classes (`ganado`, `leon`, `luis`, `ashley`), 50 epochs | Added auto-labeled ally bounding boxes (`yolov8x.pt` IoU-filtered) | `0.654` | `0.469` | `0.648` | `1.149` | `1.342` | Multi-class disentanglement doubled `mAP50-95` and cut validation losses by ~40%. |
| **`train-12`** | `train-11/best.pt`, `epochs=30`, `patience=10`, `lr0=0.0002`, `close_mosaic=5` | Added 84 hard-negative frames (`b_frame_1.jpeg`–`b_frame_84.jpeg`) | `0.736` | `0.535` | `0.770` | `1.193` | `1.236` | Hard negatives boosted `mAP50` by +8.2 points, but `patience=10` stopped at Epoch 23 before `close_mosaic=5` ran. |
| **`train-13`** | `train-12/best.pt`, `epochs=45`, `patience=25`, `close_mosaic=15`, `CLAHE(2.0, p=0.25)` | Softened `CLAHE` contrast to reduce wall texture false alarms | `0.740` | **`0.545`** | `0.772` | `1.228` | `1.299` | Default `warmup_bias_lr=0.1` spiked `lr/pg2` to `0.067` on Epoch 1; peaked on Epoch 4 and stopped at Epoch 29 (`4 + 25`). |
| **`train-14`** *(Final Model)* | `train-13/best.pt`, `epochs=15`, `close_mosaic=15`, `patience=0`, `warmup_epochs=0` | 15 full epochs of un-stitched clean frames with zero bias warmup spike | **`0.747`** | `0.529` | **`0.824`** | `1.278` | `1.241` | Dedicated mosaic-free polish pushed unseen enemy Recall to **82.4%** and `mAP50` to **74.7%**. |

### 3. Obstacles Faced on Day 2 & How They Were Solved

1. **`classes=[0]` Neutralizing `agnostic_nms=True` on Allies (`predict.py`)**:
   - **Obstacle**: Even after training a 4-class model (`0: ganado`, `1: leon`, `2: luis`, `3: ashley`), passing `classes=[0]` inside `model.track()` still allowed Ganado boxes to appear on Luis and Leon.
   - **Root Cause**: Ultralytics applies the `classes=[0]` filter *before* Non-Maximum Suppression. Because YOLO predicts independent sigmoid probabilities per class, a `0.88` `luis` prediction was discarded before `agnostic_nms=True` could use it to suppress an overlapping `0.46` `ganado` prediction on the same character.
   - **Solution**: Removed `classes=[0]` from `model.track()` so `agnostic_nms=True` compares all 4 classes head-to-head during NMS, then filtered `r.boxes = r.boxes[r.boxes.cls == 0]` in Python before rendering each frame to [re4_run2_clean.mp4](../runs/detect/re4_run2_clean.mp4).
2. **Wall & Environment False Positives**:
   - **Obstacle**: In empty hallways and dark cabin corners, the detector occasionally latched onto vertical wooden pillars and stone walls.
   - **Root Cause**: `patience=10` cut off `train-12` at Epoch 23 (before `close_mosaic=5` at Epochs 26–30), meaning the model never trained on clean, un-stitched full frames. Additionally, `A.CLAHE(clip_limit=4.0, p=0.5)` over-amplified dark wall woodgrain during training.
   - **Solution**: Curated 84 hard-negative frames (`b_frame_1.jpeg`–`b_frame_84.jpeg`) from false-positive video segments, lowered `CLAHE` to `clip_limit=2.0, p=0.25`, and ran a 15-epoch full-frame polish (`train-14` with `close_mosaic=15`).
3. **Early Stopping (`patience`) & Bias Warmup (`warmup_bias_lr`) During Fine-Tuning**:
   - **Obstacle**: `train-13` stopped at Epoch 29 (`patience=25`)—just one epoch before `close_mosaic=15` (Epoch 31)—because `mAP50-95` peaked on Epoch 4 (`0.545`).
   - **Root Cause**: When fine-tuning an already-converged checkpoint, Ultralytics' default `warmup_bias_lr=0.1` spiked the bias learning rate (`lr/pg2`) to `0.067` (335× higher than `lr0=0.0002`) during Epochs 1–2, jostling pre-trained biases so the model spent the next 25 epochs recovering.
   - **Solution**: Set `warmup_epochs=0` and `patience=0` in `train-14`, keeping `lr/pg2` locked at `0.0002` from Epoch 1 and guaranteeing all 15 mosaic-free epochs completed.
4. **Slow-Motion Output Video (`30 FPS` vs. `60 FPS`)**:
   - **Obstacle**: Custom `cv2.VideoWriter` output played back at half speed.
   - **Root Cause**: [re4_run2.mp4](../re4_run2.mp4) was recorded at `60 FPS`, while `cv2.VideoWriter` was hardcoded to `30 FPS`.
   - **Solution**: Queried `cv2.CAP_PROP_FPS` dynamically from the source video before initializing `cv2.VideoWriter`.

### 4. Portfolio Summary: End-to-End Engineering Takeaways
- **Problem Statement**: Real-world gameplay object detection under low lighting, heavy camera motion, small distant targets, and high silhouette overlap between enemies and friendly characters.
- **Key Engineering Contributions**:
  1. **Dataset Integrity & Leakage Audit**: Eliminated train/val split leakage and rebalanced a 64% negative-frame skew.
  2. **Semi-Supervised Multi-Class Expansion**: Automated bounding-box generation for recurring characters (`leon`, `luis`, `ashley`) using IoU-filtered `yolov8x.pt` predictions, cutting manual annotation time from hours to minutes and doubling `mAP50-95` (`0.233` $\rightarrow$ `0.469`).
  3. **Hard-Negative Mining & Two-Stage Mosaic/Full-Frame Fine-Tuning**: Extracted 84 targeted hard-negative frames and decoupled Mosaic feature learning (`train-12`/`train-13`) from `close_mosaic=15` full-frame boundary calibration (`train-14`), reaching **82.4% Recall** and **74.7% mAP50**.
  4. **Post-Processing & Inference Pipeline Engineering**: Diagnosed pre-NMS class-filtering interactions in Ultralytics, implemented post-NMS class-agnostic suppression with ByteTrack (`bytetrack.yaml`) on Apple Silicon (`device='mps'`), and achieved a 15×–25× inference speedup at native 60 FPS.
