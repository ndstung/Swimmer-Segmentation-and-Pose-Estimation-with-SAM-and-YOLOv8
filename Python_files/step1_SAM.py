import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sam2')))

from sam2.build_sam import build_sam2_video_predictor

# ----------------- Device Setup -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# ----------------- Config and Checkpoint -----------------
sam2_checkpoint = "C:\\267_Project\\sam2\\checkpoints\\sam2.1_hiera_large.pt"
model_cfg = "C:\\267_Project\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_l"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# # ----------------- Frame Folder -----------------
# video_folder = r"C:\267_Project\Term_project\Dataset\original_image\back-training-original"

# frame_names = [p for p in os.listdir(video_folder) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
# frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# print("Looking in:", video_folder)
# print("Folder exists:", os.path.isdir(video_folder))
# print("Number of frames found:", len(frame_names))

# # ----------------- Function for Point Selection -----------------
# def select_points_interactively(image_path, num_points=2):
#     img = Image.open(image_path)
#     plt.figure(figsize=(12, 8))
#     plt.imshow(img)
#     plt.title(f"Click {num_points} points on the swimmer, then close the window")
#     points = plt.ginput(num_points, timeout=0)
#     plt.close()
#     points_array = np.array(points, dtype=np.float32)
#     print(f"Selected Points: {points_array}")
#     return points_array

# # ----------------- User Interactive Input -----------------
# ann_frame_idx = 0  # Frame to annotate
# image_path = os.path.join(video_folder, frame_names[ann_frame_idx])
# points = select_points_interactively(image_path, num_points=3)
# labels = np.ones(points.shape[0], dtype=np.int32)

# # ----------------- Init Predictor -----------------
# inference_state = predictor.init_state(video_path=video_folder)
# predictor.reset_state(inference_state)

# _, out_obj_ids, out_mask_logits = predictor.add_new_points(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=1,
#     points=points,
#     labels=labels
# )

# # ----------------- Visualize the Annotation -----------------
# plt.figure(figsize=(12, 8))
# plt.title(f"Annotated Frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_folder, frame_names[ann_frame_idx])))

# def show_mask(mask, ax, obj_id=None, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         cmap = plt.get_cmap("tab10")
#         cmap_idx = 0 if obj_id is None else obj_id
#         color = np.array([*cmap(cmap_idx)[:3], 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)

# def show_points(coords, labels, ax, marker_size=200):
#     pos = coords[labels == 1]
#     neg = coords[labels == 0]
#     ax.scatter(pos[:, 0], pos[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg[:, 0], neg[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# show_points(points, labels, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.show()

# # ----------------- Propagate Mask Through Video -----------------
# video_segments = {}
# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }

# from PIL import Image, ImageDraw
# import numpy as np
# import os

# #----------------- Save Output Frames + Binary Masks -----------------
# # Set output folders
# output_folder = r"C:\267_Project\Term_project\Segmented_SAM_Images\back-training-sam"

# os.makedirs(output_folder, exist_ok=True)

# for out_frame_idx in range(len(frame_names)):
#     img_path = os.path.join(video_folder, frame_names[out_frame_idx])
#     img = Image.open(img_path).convert("RGBA")  # Open as RGBA

#     overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))  # Transparent layer
#     draw = ImageDraw.Draw(overlay)

#     for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
#         mask = np.squeeze(out_mask)  # Ensure it's (H, W)
#         mask_bin = (mask > 0).astype(np.uint8) * 255  # Binary mask for compositing

#         # Draw colored overlay for visualization only
#         mask_img_resized = Image.fromarray(mask_bin).resize(img.size).convert("L")
#         color = (255, 165, 0, 128)  # Orange with transparency
#         color_img = Image.new('RGBA', img.size, color)
#         overlay = Image.composite(color_img, overlay, mask_img_resized)

#     # Merge original image with overlay
#     result = Image.alpha_composite(img, overlay)
#     #result = result.resize((640, 640))

#     # Save the visual result only
#     save_path = os.path.join(output_folder, f"s{out_frame_idx + 525}.png")
#     result.save(save_path)
#     print(f"Saved {save_path}")