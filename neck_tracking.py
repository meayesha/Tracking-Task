import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# === Setup paths and directories ===
video_path = "Video/NeckDeformationClip.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

# Output root directory
output_root = "Results"
screenshot_dir = os.path.join(output_root, "Screenshots")
heatmap_dir = os.path.join(output_root, "Heatmaps")
overlay_dir = os.path.join(output_root, "Overlays")
plot_dir = os.path.join(output_root, "Plots")

# Create all directories if they don't exist
for folder in [output_root, screenshot_dir, heatmap_dir, overlay_dir, plot_dir]:
    os.makedirs(folder, exist_ok=True)

# === Video & ROI Setup ===
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

LEFT_X, LEFT_Y, LEFT_W, LEFT_H = 590, 500, 635, 520 # Adjusting the left region of the neck
RIGHT_X, RIGHT_Y, RIGHT_W, RIGHT_H = 1440, 290, 450, 670 #Adjusting the right region of the neck
frame_index = 0
left_intensity = []
right_intensity = []
frame_indices = []

# === Frame-by-frame processing ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


    # Annotate and save every 25th frame
    if frame_index % 25 == 0:

        # Generate heatmap
        heatmap = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_TURBO)
        heatmap_path = os.path.join(heatmap_dir, f"heatmap_frame_{frame_index}.jpg")
        cv2.imwrite(heatmap_path, heatmap)

        # Overlay heatmap on original
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        overlay_path = os.path.join(overlay_dir, f"overlay_frame_{frame_index}.jpg")
        cv2.imwrite(overlay_path, overlay)

        ## Get left and right ROIs
        left_roi = diff[LEFT_Y:LEFT_Y + LEFT_H, LEFT_X:LEFT_X + LEFT_W]
        right_roi = diff[RIGHT_Y:RIGHT_Y + RIGHT_H, RIGHT_X:RIGHT_X + RIGHT_W]

        # Motion intensities
        left_motion = np.sum(left_roi)
        right_motion = np.sum(right_roi)
        left_intensity.append(left_motion)
        right_intensity.append(right_motion)
        frame_indices.append(frame_index)

        # Annotate
        cv2.rectangle(overlay, (LEFT_X, LEFT_Y), (LEFT_X + LEFT_W, LEFT_Y + LEFT_H), (255, 0, 0), 2)
        cv2.putText(overlay, f"L: {int(left_motion)}", (LEFT_X, LEFT_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.rectangle(overlay, (RIGHT_X, RIGHT_Y), (RIGHT_X + RIGHT_W, RIGHT_Y + RIGHT_H), (0, 255, 0), 2)
        cv2.putText(overlay, f"R: {int(right_motion)}", (RIGHT_X, RIGHT_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(overlay, f"Frame {frame_index}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        # Save screenshot
        screenshot_path = os.path.join(screenshot_dir, f"annotated_frame_{frame_index}.jpg")
        cv2.imwrite(screenshot_path, overlay)

    prev_gray = gray.copy()
    frame_index += 1

cap.release()

# === Plot motion intensity over time ===
plt.figure(figsize=(10, 5))
plt.plot(left_intensity, label="Left Neck ROI", color='blue')
plt.plot(right_intensity, label="Right Neck ROI", color='green')
plt.title("Left vs Right Neck Motion Intensity Over Time")
plt.xlabel("Frame Index (every 25)")
plt.ylabel("Motion Intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(plot_dir, "left_right_intensity_plot.png")
plt.savefig(plot_path)
plt.show()

# === Save motion intensity data to CSV ===
csv_path = os.path.join(plot_dir, "motion_intensity_data.csv")
motion_data = {
    "Frame_Index": frame_indices,
    "Left_Motion_Intensity": left_intensity,
    "Right_Motion_Intensity": right_intensity
}
df_motion = pd.DataFrame(motion_data)
df_motion.to_csv(csv_path, index=False)


# === Summary output ===
print("\n✅ All outputs saved:")
print(f"📁 Screenshots → {screenshot_dir}")
print(f"📁 Heatmaps    → {heatmap_dir}")
print(f"📁 Overlays    → {overlay_dir}")
print(f"📁 Plots       → {plot_path}")
