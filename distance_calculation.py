import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from ultralytics.hub import hub
from ultralytics import YOLO
import cv2
import os

model_path = './best.pt'
model = YOLO(model_path)

img_path = './test.jpg'  
img = Image.open(img_path).convert("RGB")
img = np.array(img)

results = model(img_path)


masks = []

# Process results list
for result in results:
    boxe = result.boxes  # Boxes object for bounding box outputs
    mask = result.masks  # Masks object for segmentation masks outputs
    keypoint = result.keypoints  # Keypoints object for pose outputs
    prob = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.save(filename='result.jpg')  # save to disk
    masks.append(mask)
    # result.show()  # display to screen

coord = [masks[0].xy[0][:,0], masks[0].xy[0][:,1]]


# coord에서 x좌표와 y좌표 가져오기
x_coords = coord[0].tolist()
y_coords = coord[1].tolist()

# Find the extreme points
left_top_idx = np.argmin(coord[0] + coord[1])  # Index of the left top point
right_bottom_idx = np.argmax(coord[0] + coord[1])  # Index of the right bottom point
left_bottom_idx = np.argmin(coord[0] - coord[1])  # Index of the left bottom point
right_top_idx = np.argmax(coord[0] - coord[1])  # Index of the right top point

# Get the coordinates of the extreme points
left_top = (int(coord[0][left_top_idx]), int(coord[1][left_top_idx]))
right_bottom = (int(coord[0][right_bottom_idx]), int(coord[1][right_bottom_idx]))
left_bottom = (int(coord[0][left_bottom_idx]), int(coord[1][left_bottom_idx]))
right_top = (int(coord[0][right_top_idx]), int(coord[1][right_top_idx]))

# Draw circles at all coordinates
for x, y in zip(x_coords, y_coords):
    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1) 


# Draw circles at the extreme points in red
cv2.circle(img, left_top, 5, (0, 0, 255), -1)  # left top
cv2.circle(img, right_bottom, 5, (0, 0, 255), -1)  # right bottom
cv2.circle(img, left_bottom, 5, (0, 0, 255), -1)  # left bottom
cv2.circle(img, right_top, 5, (0, 0, 255), -1)  # right top

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Calculate the lengths of each side
left_right_length = abs(right_top[0] - left_top[0])
top_bottom_length = abs(left_bottom[1] - left_top[1])

# Print the lengths of each side
print("Left-Right Length:", left_right_length)
print("Top-Bottom Length:", top_bottom_length)

output_file_path = os.path.join(os.getcwd(), 'result_with_points.jpg')
cv2.imwrite(output_file_path, img)