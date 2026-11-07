# Resident Evil 4 CV (Ultralytics YOLO) — Engineering & Portfolio Log

## Session Recap: October 14–15, 2026 (Full End-of-Day Log)

### 1. Project Objective
Develop a high-precision computer vision object detection pipeline using **Ultralytics YOLOv8** on Apple Silicon (`device='mps'`) to detect enemies (**Ganados**) in *Resident Evil 4* gameplay footage ([re4_run2.mp4](../re4_run2.mp4), [re4_run3.mp4](../re4_run3.mp4), [re4_run_1080p.mov](../re4_run_1080p.mov)) across four challenging conditions:
- Dimly-lit indoor and night environments (e.g., the cabin siege).
- High false-negative / false-positive ambiguity caused by the third-person player model (Leon) and friendly NPCs (Luis, Ashley).
- Small, distant enemy silhouettes in 1080p widescreen footage.
- Non-humanoid enemy variations (Plagas mutations, Colmillos).

### 2. Progression of Experiments & Metrics Summary

| Run | Model & Config | Classes (`nc`) | Dataset Split | Peak `mAP50` | Peak `mAP50-95` | `val/box_loss` | `val/cls_loss` | Key Outcome / Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train` – `train-7`** | `yolov8n.pt`, `imgsz=640`, `batch=16` | 1 (`ganado`) | Leaked (`train` = `val`), 64% negative frames | `0.940` *(leaked)* | `0.618` *(leaked)* | `1.038` | `1.092` | High paper metrics due to train/val leakage; 587 unlabeled player frames suppressed humanoid recall in real video. |
| **`train-8`** | `yolov8s.pt`, `imgsz=960`, `rect=True`, `batch=4`, `optimizer='auto'` | 1 (`ganado`) | Split (`430` train / `135` val) | `0.002` *(collapsed)* | `0.0004` | `2.681` | `30.94` *(diverged)* | `rect=True` disabled dataset shuffling and Mosaic stitching; `auto` overrode `lr0` to `0.002`, causing BatchNorm divergence on unshuffled negative batches. |
| **`train-10`** | `yolov8s.pt`, `imgsz=768`, `rect=False`, `batch=8`, `AdamW` (`lr0=0.0005`), `CLAHE` | 1 (`ganado`) | Split (`430` train / `135` val) | `0.536` | `0.233` | `1.954` | `2.229` | Restored convergence on true unseen validation split; however, single-class detector still confused Leon, Luis, and Ashley with Ganados. |
| **`train-11`** | Fine-tuned from `train-10/best.pt`, `epochs=50`, `patience=15`, `imgsz=768`, `CLAHE` | 4 (`ganado`, `leon`, `luis`, `ashley`) | Split (`430` train / `135` val) | **`0.654`** (+22%) | **`0.469`** (+101%) | **`1.149`** (-41%) | **`1.342`** (-40%) | **Major breakthrough**: Multi-class character disentanglement cut validation box/class losses by ~40% and more than doubled `mAP50-95`. |

### 3. Obstacles Faced & Root-Cause Diagnosis

1. **Inflated Validation Metrics & Data Leakage (`train` – `train-7`)**:
   - **Symptom**: Validation `mAP50` reached `0.940–0.973` ([train-7/results.csv](../runs/detect/train-7/results.csv)), yet real video inference missed enemies in dark rooms and at a distance.
   - **Root Cause**: [data.yaml](../data.yaml) originally pointed both `train:` and `val:` to the same `images` folder, grading the model on memorized training frames.
2. **High False Negatives Caused by Excessive Negative Background Frames**:
   - **Symptom**: Real Ganados were frequently missed whenever the player or friendlies were on screen.
   - **Root Cause**: Out of 918 total frames, 587 (~64%) were unlabeled negative frames containing Leon/friendlies. Because Ultralytics recommends only 0–10% background frames, a 64% negative ratio taught the detector a strong prior that *"humanoid figure = background."*
3. **Apple Silicon Unified Memory (VRAM) Exhaustion**:
   - **Symptom**: Scaling up to `yolov8s.pt` with `imgsz=960`, `multi_scale=0.2`, `batch=16`, and `workers=8` exhausted macOS Unified Memory.
   - **Root Cause**: `batch=16` at up to `1152×1152` tripled activation memory, 8 multiprocessing DataLoader workers consumed shared CPU/GPU RAM, and `multi_scale=0.2` changed tensor dimensions every batch—fragmenting PyTorch's Metal (`mps`) buffer pool.
4. **Complete Training Collapse in `train-8` (`mAP50 = 0.00197`)**:
   - **Symptom**: In [train-8/results.csv](../runs/detect/train-8/results.csv), `val/cls_loss` spiked to `30.94` by Epoch 4 and `mAP50` collapsed to `0.000–0.002`.
   - **Root Cause**: Setting `rect=True` during training disabled dataset shuffling (`shuffle=False`) and disabled 4-image Mosaic augmentation. Combined with `batch=4` on sequential filenames (`bg_frame_*`, `cabin_frame_*`, `frame_*`) and `optimizer='auto'` silently overriding `lr0=0.001` to `0.002`, BatchNorm statistics were computed over unshuffled negative batches and diverged.
5. **400ms/Frame Video Inference Bottleneck & Ally Misdetections**:
   - **Symptom**: Running [predict.py](../predict.py) across 13,000 video frames took ~400ms/frame (~87 minutes), and the single-class `train-10` model frequently misclassified Leon, Luis, and Ashley as Ganados.
   - **Root Cause**: Omitting `device='mps'` in `model.track()` caused CPU fallback alongside 3-pass Test-Time Augmentation (`augment=True`) at `imgsz=1280`. Meanwhile, single-class training lacked explicit class boundaries to separate ally outfits/silhouettes from Ganados.

### 4. Engineering Solutions Implemented

- **Leak-Free Train/Val Split**: Separated dataset into [images/train](../images/train) (430 frames) and [images/val](../images/val) (135 frames) with matching [labels/train](../labels/train) and [labels/val](../labels/val) folders in [data.yaml](../data.yaml).
- **Low-Light & Small-Object Augmentation**: Added Albumentations `A.CLAHE(clip_limit=4.0, p=0.5)` and `hsv_v=0.4` in [train.py](../train.py) to boost contrast in dark cabin scenes, upgraded to [yolov8s.pt](../yolov8s.pt), and trained at `imgsz=768` with `scale=0.6`.
- **MPS Memory & Convergence Stabilization (`train-10`)**: Tuned `batch=8`, `workers=2`, `multi_scale=0.0` (leveraging `nbs=64` automatic gradient accumulation), set `rect=False` to restore shuffling and Mosaic stitching, and forced `optimizer='AdamW'` with `lr0=0.0005` and `cos_lr=True`.
- **Automated Multi-Class Pre-Labeling & 50-Epoch Fine-Tuning (`train-11`)**:
  - Used a COCO-pretrained `yolov8x.pt` (`person` class) script with IoU filtering (`IoU < 0.20` against existing `0: ganado` labels) to rapidly pre-label allies without re-annotating from scratch.
  - Expanded [data.yaml](../data.yaml) to 4 classes (`0: ganado`, `1: leon`, `2: luis`, `3: ashley`) and fine-tuned from `runs/detect/train-10/weights/best.pt` for 50 epochs (`patience=15`).
  - As shown in [train-11/results.csv](../runs/detect/train-11/results.csv), running the full 50 epochs (rather than stopping at 20 epochs where `mAP50` was `0.456`) allowed the newly initialized 4-class head to reach **`0.654 mAP50`**, **`0.469 mAP50-95`** (more than double `train-10`'s `0.233`), **`0.697` Precision**, and **`0.648` peak Recall**, while cutting `val/box_loss` from `2.145` to `1.158` and `val/cls_loss` from `2.321` to `1.342`.
- **15×–25× Faster Video Tracking (`predict.py`)**: Configured [predict.py](../predict.py) with `runs/detect/train-11/weights/best.pt`, `device='mps'`, `imgsz=768`, `conf=0.333`, `tracker='bytetrack.yaml'`, and `classes=[0]` to track only Ganados while ignoring Leon, Luis, and Ashley.

### 5. Key Technical Concepts Learned
- **Precision, Recall, `mAP50`, and `mAP50-95`**: Understood how `mAP50` grades coarse detection (`IoU ≥ 0.50`) while `mAP50-95` grades tight bounding-box localization (`IoU = 0.50–0.95`)—evidenced by `mAP50-95` jumping from `0.233` to `0.469` once multi-class labels resolved boundary ambiguity between overlapping characters.
- **Regularization & Loss Curves**: Learned why training loss (`train/*_loss`) remains higher than validation loss when heavy augmentations (`Mosaic`, `CLAHE`, `scale`) are active, and why `close_mosaic=10` produces a step-down in training loss during the final 10 epochs (Epochs 41–50).
- **Transfer Learning & Head Re-Initialization**: Learned why expanding from 1 class (`nc=1`) to 4 classes (`nc=4`) preserves backbone/neck feature weights from `train-10/weights/best.pt` but resets the final classification head—requiring a 50-epoch schedule (`patience=15`) rather than 20 epochs for full convergence.

### 6. Portfolio & Interview Value
- **End-to-End Iterative ML Engineering**: Shows a clear, metric-backed progression from a leaked baseline (`train-7`) $\rightarrow$ hardware/framework failure diagnosis (`train-8`) $\rightarrow$ clean single-class baseline (`train-10`, `0.536 mAP50`) $\rightarrow$ automated semi-supervised multi-class expansion (`train-11`, `0.654 mAP50`, `2× mAP50-95`).
- **Semi-Supervised Data Pipeline Design**: Built an IoU-filtered `yolov8x.pt` pre-labeling pipeline to annotate recurring characters across hundreds of frames in minutes instead of days of manual labeling.
- **Production Inference Optimization**: Combined Apple Silicon Metal (`mps`) acceleration, resolution matching (`imgsz=768`), ByteTrack Kalman filtering (`bytetrack.yaml`), and class-filtered NMS (`classes=[0]`) to process 13,000-frame gameplay videos efficiently.
