import cv2
import numpy as np
import os

# === VIDEO PATH ===
video_path = "Video/NeckDeformationClip.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError("❌ Cannot open video.")

# === ROI COORDINATES ===
LEFT_X, LEFT_Y, LEFT_W, LEFT_H = 590, 500, 635, 520
RIGHT_X, RIGHT_Y, RIGHT_W, RIGHT_H = 1440, 290, 450, 670

# === SETUP VIDEO WRITER ===
ret, first_frame = cap.read()
if not ret:
    raise IOError("❌ Cannot read first frame.")

height, width, _ = first_frame.shape
output_path = "Results/Annotated_Live_Video.mp4"
os.makedirs("Results", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, 20, (width, height))  # 20 FPS

# First frame preprocessing
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
frame_index = 0

# === PROCESS FRAMES ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # === ROI MOTION INTENSITY ===
    left_roi = diff[LEFT_Y:LEFT_Y + LEFT_H, LEFT_X:LEFT_X + LEFT_W]
    right_roi = diff[RIGHT_Y:RIGHT_Y + RIGHT_H, RIGHT_X:RIGHT_X + RIGHT_W]
    left_motion = np.sum(left_roi)
    right_motion = np.sum(right_roi)

    # === HEATMAP OVERLAY ===
    heatmap = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # === DRAW ANNOTATIONS ===
    cv2.rectangle(overlay, (LEFT_X, LEFT_Y), (LEFT_X + LEFT_W, LEFT_Y + LEFT_H), (255, 0, 0), 2)
    cv2.putText(overlay, f"Left: {int(left_motion)}", (LEFT_X, LEFT_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.rectangle(overlay, (RIGHT_X, RIGHT_Y), (RIGHT_X + RIGHT_W, RIGHT_Y + RIGHT_H), (0, 255, 0), 2)
    cv2.putText(overlay, f"Right: {int(right_motion)}", (RIGHT_X, RIGHT_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(overlay, f"Frame {frame_index}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # === SHOW + SAVE ===
    cv2.imshow("Live Annotated Video", overlay)
    video_writer.write(overlay)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()
    frame_index += 1

# === CLEANUP ===
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"\n✅ Annotated video saved at: {output_path}")
