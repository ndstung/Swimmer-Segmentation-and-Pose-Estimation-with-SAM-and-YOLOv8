import os
import json

# --- Configuration ---
IMAGE_FOLDER     = r"C:\267_Project\Term_project\Joint_Estimation_SAM_Images\freestyle-training-pose-out"                        # Folder with original images
OLD_JSON_PATH    = r"C:\267_Project\Term_project\Joint_Estimation_SAM_Images\freestyle-training-pose-out\joint_coordinates.json" # Original JSON file
NEW_JSON_PATH    = r"C:\267_Project\Term_project\freestyle_joint_coordinates_update.json"    # Output path
RENAME_PREFIX    = "freestyle_"
EXTENSIONS       = {'.jpg', '.jpeg', '.png'}
SAVE_RENAME_LOG  = True  # Set to False if you don't need a CSV log

# --- Load JSON ---
with open(OLD_JSON_PATH, 'r') as f:
    data = json.load(f)

# --- Collect and Sort Image Files ---
image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if os.path.splitext(f)[1].lower() in EXTENSIONS
])

# --- Rename Images and Build Mapping ---
updated_data = {}
rename_log = []

if len(image_files) != len(data):
    raise ValueError("⚠️ Image count and JSON entry count do not match.")

for idx, (old_key, old_fname) in enumerate(zip(sorted(data.keys()), image_files), 1):
    ext = os.path.splitext(old_fname)[1].lower()
    new_name = f"{RENAME_PREFIX}{idx:04d}{ext}"
    old_path = os.path.join(IMAGE_FOLDER, old_fname)
    new_path = os.path.join(IMAGE_FOLDER, new_name)

    os.rename(old_path, new_path)
    updated_data[new_name] = data[old_key]
    rename_log.append((old_fname, new_name))

    print(f"Renamed: {old_fname} → {new_name}")

# --- Save Updated JSON ---
with open(NEW_JSON_PATH, 'w') as f:
    json.dump(updated_data, f, indent=2)

print(f"\n✅ Updated JSON saved to: {NEW_JSON_PATH}")   