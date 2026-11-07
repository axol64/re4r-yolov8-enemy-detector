from pathlib import Path
from ultralytics import YOLO

# Load COCO-pretrained YOLOv8x (COCO class 0 is 'person')
coco_model = YOLO('yolov8x.pt')

Class_Names = ['ganado', 'leon', 'luis', 'ashley']


def compute_iou(box1, box2):
  """Calculate IoU between two normalized YOLO boxes [x_center, y_center, w, h]."""
  b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
  b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
  b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
  b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

  inter_w = max(0.0, min(b1_x2, b2_x2) - max(b1_x1, b2_x1))
  inter_h = max(0.0, min(b1_y2, b2_y2) - max(b1_y1, b2_y1))
  inter_area = inter_w * inter_h
  union_area = (box1[2] * box1[3]) + (box2[2] * box2[3]) - inter_area
  return inter_area / union_area if union_area > 0 else 0.0


for split in ['train', 'val']:
  img_dir = Path('images') / split
  lbl_dir = Path('labels') / split
  out_dir = Path('labels_multiclass') / split
  out_dir.mkdir(parents=True, exist_ok=True)

  # Save classes.txt so annotation tools (LabelImg, AnyLabeling, MakeSense) load class names
  (out_dir / 'classes.txt').write_text('\n'.join(Class_Names) + '\n')

  img_paths = sorted(img_dir.glob('*.jpg'))
  print(f'Pre-labeling {len(img_paths)} images in {split}...')

  for img_path in img_paths:
    lbl_path = lbl_dir / f'{img_path.stem}.txt'
    existing_lines = (
        lbl_path.read_text().strip().splitlines() if lbl_path.exists() else []
    )
    ganado_boxes = [
        [float(x) for x in line.split()[1:5]]
        for line in existing_lines
        if line.strip()
    ]

    # Detect human figures on Apple Silicon GPU
    preds = coco_model.predict(
        source=str(img_path),
        classes=[0],  # COCO class 0 = person
        conf=0.40,
        device='mps',
        verbose=False,
    )[0]

    new_lines = list(existing_lines)
    for box in preds.boxes.xywhn.cpu().tolist():
      # Only add if the detected person does NOT overlap an already-labeled Ganado
      if all(compute_iou(box, g_box) < 0.20 for g_box in ganado_boxes):
        # Default to class 1 (leon); you can quickly toggle Luis (2) or Ashley (3) in your label tool
        new_lines.append(
            f'1 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}'
        )

    out_file = out_dir / f'{img_path.stem}.txt'
    out_file.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''))

print('Done! Check labels_multiclass/train and labels_multiclass/val.')
