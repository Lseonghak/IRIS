import cv2
import json
import numpy as np

def apply_mask(image_path, mask_json_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Read the JSON file
    with open(mask_json_path, 'r') as f:
        mask_data = json.load(f)
    
    # Create an empty mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Draw the polygons specified in the JSON file on the mask
    for shape in mask_data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    
    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    # Save the result
    cv2.imwrite(output_path, masked_img)
    print(f"Masked image saved at: {output_path}")

# Paths to the input files and output
image_path = '/home/seonghak/ultralytics/stereo/a.jpg'
mask_json_path = '/home/seonghak/ultralytics/stereo/a.json'
output_path = '/home/seonghak/ultralytics/stereo/a_masked_image.jpg'

apply_mask(image_path, mask_json_path, output_path)
