import albumentations as A
from ultralytics import YOLO

model = YOLO('runs/detect/train-13/weights/best.pt')

custom_transforms = [
    A.CLAHE(clip_limit=2.0, p=0.25)
                     ]

print("Initiating training sequence on Apple Silicon...")
results = model.train(
    data='data.yaml',
    epochs=15,
    warmup_epochs=0,
    imgsz=768,
    scale=0.6,
    batch=8,
    workers=2,
    device='mps',
    optimizer='AdamW',
    lr0=0.0002,
    cos_lr=True,
    hsv_v=0.4,
    close_mosaic=15,
    augmentations=custom_transforms,
    resume=False,
    patience=0
    )
