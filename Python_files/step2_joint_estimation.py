import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- Configuration ----------------
INPUT_FOLDER     = r"C:\267_Project\Term_project\Segmented_SAM_Images\back-training-sam"
OUTPUT_FOLDER    = r"C:\267_Project\Term_project\Joint_Estimation_SAM_Images\back-training-pose-out"
CONF_THRESHOLD   = 0.2
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
model = YOLO('yolov8m-pose.pt')

coordinates = {}

# Define COCO keypoint skeleton and assign colors to limbs (BGR format)
colored_skeleton = [
    ((5, 7),  (255, 0, 255)),   # Left arm - pink
    ((7, 9),  (255, 0, 255)),   # Left forearm - pink
    ((6, 8),  (0, 255, 0)),     # Right arm - green
    ((8,10),  (0, 255, 0)),     # Right forearm - green
    ((5, 6),  (255, 0, 0)),     # Shoulders - blue
    ((11,13), (0, 165, 255)),   # Left thigh - orange
    ((13,15), (0, 165, 255)),   # Left calf - orange
    ((12,14), (255, 255, 0)),   # Right thigh - light blue
    ((14,16), (255, 255, 0)),   # Right calf - light blue
    ((11,12), (128, 0, 128)),   # Hips - purple
    ((5,11),  (128, 0, 128)),   # Left torso - purple
    ((6,12),  (128, 0, 128))    # Right torso - purple
]

# # Simplified skeleton (left/right limbs, core body)
# colored_skeleton = [
#     ((5, 7),  (255, 0, 255)),   # Left upper arm
#     ((6, 8),  (0, 255, 0)),     # Right upper arm
#     ((11,13), (0, 165, 255)),   # Left thigh
#     ((12,14), (255, 255, 0)),   # Right thigh
#     ((5, 6),  (255, 0, 0)),     # Shoulders
#     ((11,12), (128, 0, 128))    # Hips
# ]

for fname in os.listdir(INPUT_FOLDER):
    stem, ext = os.path.splitext(fname)
    if ext.lower() not in VALID_EXTENSIONS:
        continue

    # load & run
    img_bgr = cv2.imread(os.path.join(INPUT_FOLDER, fname))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res     = model(img_rgb, conf=CONF_THRESHOLD)[0]

    # get boxes, scores, keypoints
    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    kpts = res.keypoints.xy.cpu().numpy()

    if scores.size == 0:
        print(f"No person in {fname}")
        continue

    # pick highest-confidence person
    best_i = int(np.argmax(scores))
    best_box  = boxes[best_i].astype(int)
    best_kpts = kpts[best_i].astype(int)
    best_score = scores[best_i]

    # record that swimmer’s joints
    coordinates[fname] = best_kpts.tolist()

    # draw on a copy
    out_img = img_bgr.copy()
    x1, y1, x2, y2 = best_box
    cv2.rectangle(out_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # Draw label background (blue box with white text)
    label_text = f"person {best_score:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(out_img, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1), (255, 0, 0), -1)  # filled blue rectangle
    cv2.putText(out_img,
                label_text,
                (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1,
                cv2.LINE_AA)

    # draw keypoints
    for (x, y) in best_kpts:
        cv2.circle(out_img, (x, y), 4, (0, 0, 255), -1)

    # # draw colored skeleton lines
    for (i, j), color in colored_skeleton:
        pt1 = tuple(best_kpts[i])
        pt2 = tuple(best_kpts[j])
        if (pt1 != (0, 0)) and (pt2 != (0, 0)):
            cv2.line(out_img, pt1, pt2, color, 2)

    # save
    out_path = os.path.join(OUTPUT_FOLDER, f"{stem}_pose.jpg")
    cv2.imwrite(out_path, out_img)

# finally dump JSON of only the positives
with open(os.path.join(OUTPUT_FOLDER, "joint_coordinates.json"), 'w') as f:
    json.dump(coordinates, f, indent=2)

print(f"Saved {len(coordinates)} images → {OUTPUT_FOLDER}")

