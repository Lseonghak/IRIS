import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from ultralytics.hub import hub
from ultralytics import YOLO
import cv2
import os
import time


model_path = './best.pt'
model = YOLO(model_path)

video_path = './paul.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

output_path = './output_paul.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

total_frames = 0
start_time = time.time() 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        if result.masks is not None:  
            for mask in result.masks:
                if mask is not None and hasattr(mask, 'xy') and mask.xy is not None:
                    coords = mask.xy[0]
                    for coord in coords:
                        cv2.circle(frame, (int(coord[0]), int(coord[1])), 3, (0, 255, 0), -1)  # 녹색 점

                    leftmost = min(coords, key=lambda x: x[0])
                    rightmost = max(coords, key=lambda x: x[0])
                    topmost = min(coords, key=lambda x: x[1])
                    bottommost = max(coords, key=lambda x: x[1])

                    left_right_length = np.abs(rightmost[0] - leftmost[0])
                    top_bottom_length = np.abs(bottommost[1] - topmost[1])

                    cv2.putText(frame, f"Left-Right Length: {left_right_length}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Top-Bottom Length: {top_bottom_length}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(frame)
    total_frames += 1


cap.release()
out.release()
# cv2.destroyAllWindows()

end_time = time.time() 
total_time = end_time - start_time 
fps = total_frames / total_time  

print(f"Video processing completed. Output saved to: {output_path}")
print(f"Total frames processed: {total_frames}")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Calculated FPS: {fps:.2f}")