import os
import json
import pandas as pd

# --- Configuration ---
INPUT_FOLDER = r"C:\267_Project\Term_project"  # ← Replace this
OUTPUT_CSV = r"C:\267_Project\Term_project\all_strokes_joint_data.csv"

# Define mapping: filename keyword → stroke type
stroke_map = {
    "freestyle": "Freestyle",
    "back": "Backstroke",
    "breast": "Breaststroke",
    "fly": "Butterfly"
}

# --- Process JSON Files ---
combined_rows = []

for fname in os.listdir(INPUT_FOLDER):
    if not fname.endswith(".json"):
        continue

    stroke_type = None
    for keyword, stroke in stroke_map.items():
        if keyword in fname.lower():
            stroke_type = stroke
            break

    if not stroke_type:
        print(f"⚠️ Skipping unrecognized file: {fname}")
        continue

    json_path = os.path.join(INPUT_FOLDER, fname)
    with open(json_path, "r") as f:
        data = json.load(f)

    for frame, keypoints in data.items():
        row = {"Frame": frame, "Stroke_Type": stroke_type}
        for i, (x, y) in enumerate(keypoints):
            row[f"Joint{i}_x"] = x
            row[f"Joint{i}_y"] = y
        combined_rows.append(row)

# --- Create and Save CSV ---
df = pd.DataFrame(combined_rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Combined CSV saved to: {OUTPUT_CSV}")
