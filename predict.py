import cv2
from ultralytics import YOLO

source_video = 're4_run2.mp4'
cap = cv2.VideoCapture(source_video)
fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
cap.release()

model = YOLO('runs/detect/train-14/weights/best.pt')

results = model.track(
    source=source_video,
    save=False,
    imgsz=768,
    conf=0.45,
    iou=0.50,
    agnostic_nms=True,        # Now compares all 4 classes because classes=[0] is removed below!
    tracker='bytetrack.yaml',
    stream=True,
    device='mps'
)

writer = None
for r in results:
    # Filter to class 0 (ganado) AFTER agnostic_nms has already used class 2 (luis) to suppress false ganado boxes
    r.boxes = r.boxes[r.boxes.cls == 0]
    frame = r.plot()
    if writer is None:
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter('runs/detect/re4_run2_clean.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    writer.write(frame)

if writer:
    writer.release()
print("Saved to runs/detect/re4_run2_clean.mp4")
