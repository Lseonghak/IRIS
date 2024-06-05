import cv2
import numpy as np
from ultralytics import YOLO
import math

# Load the model
model_path = './checkpoints/best.pt'
model = YOLO(model_path)

# Load stereo images
imgL_path = './data/left.png'
imgR_path = './data/right.png'
imgL = cv2.imread(imgL_path)
imgR = cv2.imread(imgR_path)

# Convert images to grayscale
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Stereo matching parameters
num_disparities = 16 * 10  # The disparity search range (must be divisible by 16)
block_size = 15  # The size of the block window to match

# Create StereoBM object
stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

# Compute disparity map
disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

focal_length = 28  # mm
baseline = 0.15  # meters

# Calculate depth map
depth_map = (focal_length * baseline) / (disparity + 0.1)  # Avoid division by zero

# Normalize depth map for visualization
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map_normalized = np.uint8(depth_map_normalized)

# Save the depth map
depth_map_path = imgL_path.replace('.png', '_depth_map.png')
cv2.imwrite(depth_map_path, depth_map_normalized)
print(f"Depth map saved to {depth_map_path}")

# Perform object detection
results = model(imgL)

# Check if there are any detections
if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
    # Get the mask from the results
    mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (imgL.shape[1], imgL.shape[0]))

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    overlay = imgL.copy()
    for contour in contours:
        for point in contour:
            cv2.circle(overlay, tuple(point[0]), 1, (0, 255, 0), -1)  # Draw points for the mask outline

    # Initialize extreme points
    leftmost = tuple(contours[0][0][0])
    rightmost = tuple(contours[0][0][0])

    for contour in contours:
        for point in contour:
            if point[0][0] < leftmost[0]:
                leftmost = tuple(point[0])
            if point[0][0] > rightmost[0]:
                rightmost = tuple(point[0])

    # Calculate the line equation for the line connecting leftmost and rightmost points
    dx = rightmost[0] - leftmost[0]
    dy = rightmost[1] - leftmost[1]
    if dx != 0:
        slope = dy / dx
        intercept = leftmost[1] - slope * leftmost[0]
    else:
        slope = float('inf')
        intercept = leftmost[0]

    # Find the maximum height difference perpendicular to the line
    max_height_diff = 0
    min_y_at_max_diff = None
    max_y_at_max_diff = None

    for contour in contours:
        for point in contour:
            if dx != 0:
                # Calculate the perpendicular distance from the point to the line
                x, y = point[0]
                perp_slope = -1 / slope
                perp_intercept = y - perp_slope * x
                intersect_x = (perp_intercept - intercept) / (slope - perp_slope)
                intersect_y = slope * intersect_x + intercept
                height_diff = np.abs(y - intersect_y)
            else:
                # If the line is vertical
                height_diff = np.abs(point[0][0] - intercept)

            if height_diff > max_height_diff:
                max_height_diff = height_diff
                min_y_at_max_diff = (x, int(intersect_y))
                max_y_at_max_diff = (x, y)

    joint_width_pixels = rightmost[0] - leftmost[0]
    height_deflection_pixels = max_height_diff

    # Calculate pixel to meter ratio
    disparity_width = disparity.shape[1]
    pixel_to_meter_ratio = (baseline / focal_length) * (imgL.shape[1] / disparity_width)
    
    # Convert deflections to meters
    width_deflection = joint_width_pixels * pixel_to_meter_ratio
    height_deflection = height_deflection_pixels * pixel_to_meter_ratio
    
    print("width pixels:", joint_width_pixels)
    print("height pixels:", height_deflection_pixels)
    print("disparity width:", disparity_width)
    print("pixel-to-meter:", pixel_to_meter_ratio)

    # Calculate the angle between the two lines
    if dx != 0:
        angle = math.degrees(math.atan(perp_slope) - math.atan(slope))
    else:
        angle = 90.0  # If the line is vertical, the angle is 90 degrees

    # Add height, width deflection, and angle text
    cv2.putText(overlay, f"Height: {height_deflection:.2f} m", (leftmost[0], leftmost[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.putText(overlay, f"Width: {width_deflection:.2f} m", (leftmost[0], leftmost[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.putText(overlay, f"Angle: {angle:.2f} degrees", (leftmost[0], leftmost[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Visualize the extreme points and the points defining the height deflection
    cv2.circle(overlay, leftmost, 5, (255, 0, 0), -1)  # Blue for leftmost
    cv2.circle(overlay, rightmost, 5, (0, 0, 255), -1)  # Red for rightmost
    cv2.circle(overlay, min_y_at_max_diff, 5, (0, 255, 255), -1)  # Yellow for min_y_at_max_diff
    cv2.circle(overlay, max_y_at_max_diff, 5, (255, 0, 255), -1)  # Magenta for max_y_at_max_diff

    # Draw line connecting leftmost and rightmost points
    cv2.line(overlay, leftmost, rightmost, (0, 255, 0), 2)  # Green line

    # Draw line for max height deflection
    if min_y_at_max_diff and max_y_at_max_diff:
        cv2.line(overlay, min_y_at_max_diff, max_y_at_max_diff, (255, 255, 0), 2)  # Cyan line for max height deflection

    # Save the result
    output_path = imgL_path.replace('.png', '_deflection.png')
    cv2.imwrite(output_path, overlay)
    print(f"Deflection analysis results saved to {output_path}")
    print(f"Angle between the lines: {angle:.2f} degrees")
else:
    print("No deflections detected in the image.")
