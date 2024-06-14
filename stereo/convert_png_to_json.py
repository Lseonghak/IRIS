import cv2
import json
import numpy as np

# Load the image
img_path = "/home/seonghak/ultralytics/stereo/a.png"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to get the binary mask
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract points from contours
points = []
for contour in contours:
    for point in contour:
        points.append([float(point[0][0]), float(point[0][1])])

# Initialize the JSON structure
json_data = {
    "version": "5.4.1",
    "flags": {},
    "shapes": [
        {
            "label": "joint deflection",
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        }
    ],
    "imagePath": "a.png",
    "imageHeight": img.shape[0],
    "imageWidth": img.shape[1]
}

# Save to JSON file
json_path = "/home/seonghak/ultralytics/stereo/a.json"
with open(json_path, "w") as json_file:
    json.dump(json_data, json_file, indent=2)

print(f"JSON file saved at: {json_path}")
