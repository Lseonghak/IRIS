import cv2
import numpy as np
from ultralytics import YOLO
import math
import time

# Load the model
model_path = './best.pt'
model = YOLO(model_path)

# Load the videos
videoL_path = './left_videos/0604_left_2.mp4'
videoR_path = './right_videos/0604_right_2.mp4'
capL = cv2.VideoCapture(videoL_path)
capR = cv2.VideoCapture(videoR_path)

if not capL.isOpened() or not capR.isOpened():
    print("Error: Cannot open videos.")
    exit()

# Output video path and settings
output_path = './0604_processed_2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(capL.get(3)), int(capL.get(4))))

total_frames = 0
detected_frames = 0
start_time = time.time()

focal_length = 28  # mm
baseline = 0.15  # meters

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    # Convert images to grayscale
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Stereo matching parameters
    num_disparities = 16 * 10  # The disparity search range (must be divisible by 16)
    block_size = 15  # The size of the block window to match

    # Create StereoBM object
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    # Compute disparity map
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Calculate depth map
    depth_map = (focal_length * baseline) / (disparity + 0.1)  # Avoid division by zero

    # Perform object detection
    results = model(frameL)

    frame_has_detections = False

    for result in results:
        if result.masks is not None:
            for mask in result.masks:
                if mask is not None and hasattr(mask, 'xy') and mask.xy is not None:
                    coords = mask.xy[0]
                    if len(coords) == 0:
                        continue  # Skip if no coordinates are available

                    # Initialize extreme points
                    leftmost = min(coords, key=lambda x: x[0])
                    rightmost = max(coords, key=lambda x: x[0])

                    # Ensure coordinates are integers
                    leftmost = (int(leftmost[0]), int(leftmost[1]))
                    rightmost = (int(rightmost[0]), int(rightmost[1]))

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

                    for coord in coords:
                        x, y = coord
                        x, y = int(x), int(y)
                        if dx != 0 and slope != 0:
                            # Calculate the perpendicular distance from the point to the line
                            perp_slope = -1 / slope
                            perp_intercept = y - perp_slope * x
                            intersect_x = (perp_intercept - intercept) / (slope - perp_slope)
                            intersect_y = slope * intersect_x + intercept
                            height_diff = np.abs(y - intersect_y)
                        elif slope == 0:
                            # If the line is horizontal
                            height_diff = np.abs(y - intercept)
                        else:
                            # If the line is vertical
                            height_diff = np.abs(x - intercept)

                        if height_diff > max_height_diff:
                            max_height_diff = height_diff
                            min_y_at_max_diff = (x, int(intersect_y)) if slope != 0 else (x, y - int(height_diff))
                            max_y_at_max_diff = (x, y)

                    joint_width_pixels = rightmost[0] - leftmost[0]
                    height_deflection_pixels = max_height_diff

                    # Calculate pixel to meter ratio
                    disparity_width = frameL.shape[1]  # Assuming the frame width is the same as the image width
                    pixel_to_meter_ratio = (baseline / focal_length) * (frameL.shape[1] / disparity_width)

                    # Convert deflections to meters
                    width_deflection = joint_width_pixels * pixel_to_meter_ratio
                    height_deflection = height_deflection_pixels * pixel_to_meter_ratio

                    # Calculate the angle between the two lines
                    if dx != 0 and slope != 0:
                        angle = math.degrees(math.atan(perp_slope) - math.atan(slope))
                    else:
                        angle = 90.0  # If the line is vertical or horizontal, the angle is 90 degrees

                    # Add height, width deflection, and angle text
                    cv2.putText(frameL, f"Height: {height_deflection:.2f} m", (leftmost[0], leftmost[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.putText(frameL, f"Width: {width_deflection:.2f} m", (leftmost[0], leftmost[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.putText(frameL, f"Angle: {angle:.2f} degrees", (leftmost[0], leftmost[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    # Visualize the extreme points and the points defining the height deflection
                    cv2.circle(frameL, leftmost, 5, (255, 0, 0), -1)  # Blue for leftmost
                    cv2.circle(frameL, rightmost, 5, (0, 0, 255), -1)  # Red for rightmost
                    if min_y_at_max_diff and max_y_at_max_diff:
                        cv2.circle(frameL, min_y_at_max_diff, 5, (0, 255, 255), -1)  # Yellow for min_y_at_max_diff
                        cv2.circle(frameL, max_y_at_max_diff, 5, (255, 0, 255), -1)  # Magenta for max_y_at_max_diff

                    # Draw line connecting leftmost and rightmost points
                    cv2.line(frameL, leftmost, rightmost, (0, 255, 0), 2)  # Green line

                    # Draw line for max height deflection
                    if min_y_at_max_diff and max_y_at_max_diff:
                        cv2.line(frameL, min_y_at_max_diff, max_y_at_max_diff, (255, 255, 0), 2)  # Cyan line for max height deflection

                    frame_has_detections = True

    if frame_has_detections:
        detected_frames += 1

    out.write(frameL)
    total_frames += 1

capL.release()
capR.release()
out.release()

end_time = time.time()
total_time = end_time - start_time
fps = total_frames / total_time

print(f"Video processing completed. Output saved to: {output_path}")
print(f"Total frames processed: {total_frames}")
print(f"Total frames with detections: {detected_frames}")
print(f"Calculated FPS: {fps:.2f}")
