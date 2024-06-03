import cv2
import math
import numpy as np
from ultralytics import YOLO

# Load the model
model_path = './best.pt'
model = YOLO(model_path)

# Load the image
img_path = 'images/ex3.jpg'
img = cv2.imread(img_path)

# Get image dimensions
h, w, _ = img.shape
pixel_per_meter = 765  # Default value, you can calibrate this by hand

# Colors for visualization
txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

# Perform segmentation
results = model(img_path)

# Ensure there are at least two results
if results:
    result = results[0]
    if result.boxes and len(result.boxes.cls) >= 1:
        masks = result.masks
        if masks is not None and len(masks.xy) > 0:
            mask = masks.data[0].cpu().numpy().astype(np.uint8)  # Get the first mask and convert it to uint8
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # Resize the mask to match the image dimensions

            # Visualize the segmentation mask for the segmentation results file
            color_mask = np.zeros_like(img)  # Create a blank mask with the same dimensions as the image
            color = (0, 255, 0)  # Choose the color for the mask (e.g., green)
            color_mask[mask == 1] = color  # Set the color for the masked region
            img_with_segmentation = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)  # Blend the image with the color mask

            # Find contours in the mask and draw them on the segmentation result
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(img_with_segmentation, [largest_contour], -1, (0, 0, 255), 2)  # Red contour

                # Save the segmentation results with contours
                output_segmentation_path = img_path.replace('.jpg', '_segmentation.jpg')
                cv2.imwrite(output_segmentation_path, img_with_segmentation)
                print(f"Segmentation results saved to {output_segmentation_path}")

                # Proceed with the height and width annotations without drawing contours
                edge_pixels = largest_contour.squeeze(axis=1)
                x_coords = edge_pixels[:, 0]
                y_coords = edge_pixels[:, 1]

                # Visualize the segmentation mask for the height/width results file without contours
                img_with_mask = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)  # Blend the image with the color mask

                # Continue with the rest of the processing
                left_top_idx = np.argmin(x_coords + y_coords)
                right_bottom_idx = np.argmax(x_coords + y_coords)
                left_bottom_idx = np.argmin(x_coords - y_coords)
                right_top_idx = np.argmax(x_coords - y_coords)

                left_top = (int(x_coords[left_top_idx]), int(y_coords[left_top_idx]))
                right_bottom = (int(x_coords[right_bottom_idx]), int(y_coords[right_bottom_idx]))
                left_bottom = (int(x_coords[left_bottom_idx]), int(y_coords[left_bottom_idx]))
                right_top = (int(x_coords[right_top_idx]), int(y_coords[right_top_idx]))

                # Calculate midpoints
                left_mid = ((left_top[0] + left_bottom[0]) // 2, (left_top[1] + left_bottom[1]) // 2)
                right_mid = ((right_top[0] + right_bottom[0]) // 2, (right_top[1] + right_bottom[1]) // 2)

                # Draw a line along the length of the joint deflection
                cv2.line(img_with_mask, left_mid, right_mid, (0, 0, 255), 2)

                # Calculate the length of the joint deflection
                x1, y1 = left_mid
                x2, y2 = right_mid
                length = (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) / pixel_per_meter

                # Method 1: Absolute height
                max_height = 0
                max_height_points = None

                unique_x = np.unique(x_coords)
                for x in unique_x:
                    y_values = y_coords[x_coords == x]
                    if len(y_values) >= 2:
                        height = (np.max(y_values) - np.min(y_values))
                        if height > max_height:
                            max_height = height
                            max_height_points = ((int(x), int(np.min(y_values))), (int(x), int(np.max(y_values))))

                if max_height_points:
                    cv2.line(img_with_mask, max_height_points[0], max_height_points[1], (255, 0, 0), 2)
                    cv2.putText(img_with_mask, f"Height: {max_height/pixel_per_meter:.2f} m", 
                                (max_height_points[0][0], max_height_points[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, txt_color, 2)

                # Display the length and height on the image with a consistent font size and position
                font_scale = 0.9
                thickness = 2
                text_length = f"Length: {length:.2f} m"
                text_height = f"Height: {max_height/pixel_per_meter:.2f} m"

                text_size_length, _ = cv2.getTextSize(text_length, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_size_height, _ = cv2.getTextSize(text_height, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                # Position for length text
                text_x_length, text_y_length = (left_mid[0], left_mid[1] - 10)
                cv2.putText(img_with_mask, text_length, (text_x_length, text_y_length),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, thickness)

                # Position for height text
                text_x_height, text_y_height = (max_height_points[0][0], max_height_points[0][1] - 10)
                cv2.putText(img_with_mask, text_height, (text_x_height, text_y_height),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, thickness)

                # Save the final image with both annotations
                output_file_path = img_path.replace('.jpg', '_height_width.jpg')
                cv2.imwrite(output_file_path, img_with_mask)
                print(f"Output saved to {output_file_path}")

            else:
                print("No contours found in the mask.")
        else:
            print("The first joint deflection does not contain masks.")
    else:
        print("No joint deflections detected in the image.")
else:
    print("No results detected in the image.")
