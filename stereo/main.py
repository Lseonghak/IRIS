import json
import cv2
import numpy as np
import os
import math

def get_camera_params(stereo_params):
    camera_params1 = stereo_params['CameraParameters1']
    camera_params2 = stereo_params['CameraParameters2']
    pose_camera2 = stereo_params['PoseCamera2']

    intrinsic_matrix1 = np.array(camera_params1['K'])
    intrinsic_matrix2 = np.array(camera_params2['K'])
    radial_distortion1 = np.array(camera_params1['RadialDistortion'])
    radial_distortion2 = np.array(camera_params2['RadialDistortion'])

    dist_coeffs1 = np.zeros(4)
    dist_coeffs1[:len(radial_distortion1)] = radial_distortion1

    dist_coeffs2 = np.zeros(4)
    dist_coeffs2[:len(radial_distortion2)] = radial_distortion2

    rotation_matrix = np.array(pose_camera2['R'])
    translation_vector = np.array(pose_camera2['Translation'])

    return intrinsic_matrix1, intrinsic_matrix2, dist_coeffs1, dist_coeffs2, rotation_matrix, translation_vector

def load_images(left_img_path, right_img_path):
    imgL = cv2.imread(left_img_path)
    imgR = cv2.imread(right_img_path)
    return imgL, imgR

def stereo_rectification(camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, img_shape, R, T):
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, img_shape, R, T)
    return R1, R2, P1, P2

def compute_disparity(rectified_imgL, rectified_imgR):
    num_disparities = 16 * 10
    block_size = 7
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(cv2.cvtColor(rectified_imgL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rectified_imgR, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
    return disparity

def create_mask(image_shape, points):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask

def find_extreme_points(contours):
    leftmost = tuple(contours[0][0][0])
    rightmost = tuple(contours[0][0][0])
    for contour in contours:
        for point in contour:
            if point[0][0] < leftmost[0]:
                leftmost = tuple(point[0])
            if point[0][0] > rightmost[0]:
                rightmost = tuple(point[0])
    return leftmost, rightmost

def calculate_line_equation(leftmost, rightmost):
    dx = rightmost[0] - leftmost[0]
    dy = rightmost[1] - leftmost[1]
    if dx != 0:
        slope = dy / dx
        intercept = leftmost[1] - slope * leftmost[0]
    else:
        slope = float('inf')
        intercept = leftmost[0]
    return slope, intercept

def calculate_max_height_diff(contours, slope, intercept, dx):
    max_height_diff = 0
    min_y_at_max_diff = None
    max_y_at_max_diff = None

    for contour in contours:
        for point in contour:
            if dx != 0:
                x, y = point[0]
                perp_slope = -1 / slope
                perp_intercept = y - perp_slope * x
                intersect_x = (perp_intercept - intercept) / (slope - perp_slope)
                intersect_y = slope * intersect_x + intercept
                height_diff = np.abs(y - intersect_y)
            else:
                height_diff = np.abs(point[0][0] - intercept)

            if height_diff > max_height_diff:
                max_height_diff = height_diff
                min_y_at_max_diff = (x, int(intersect_y))
                max_y_at_max_diff = (x, y)

    return max_height_diff, min_y_at_max_diff, max_y_at_max_diff

def calculate_deflection_pixels(leftmost, rightmost, max_height_diff):
    joint_width_pixels = rightmost[0] - leftmost[0]
    height_deflection_pixels = max_height_diff
    return joint_width_pixels, height_deflection_pixels

def calculate_pixel_to_meter_ratio(baseline, focal_length, img_shape, disparity_width):
    return (baseline / focal_length) * (img_shape[1] / disparity_width)

def convert_to_mm(joint_width_pixels, height_deflection_pixels, pixel_to_meter_ratio):
    width_deflection_mm = joint_width_pixels * pixel_to_meter_ratio * 10
    height_deflection_mm = height_deflection_pixels * pixel_to_meter_ratio * 10  
    return width_deflection_mm, height_deflection_mm

def calculate_angle(dx, slope):
    if dx != 0:
        angle = math.degrees(math.atan(-1 / slope) - math.atan(slope))
    else:
        angle = 90.0
    return angle

def annotate_image(overlay, leftmost, rightmost, min_y_at_max_diff, max_y_at_max_diff, height_deflection_mm, width_deflection_mm):
    font_scale = 2
    font_thickness = 5
    cv2.putText(overlay, f"Height: {height_deflection_mm:.2f} mm", (leftmost[0], leftmost[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(overlay, f"Width: {width_deflection_mm:.2f} mm", (leftmost[0], leftmost[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    cv2.circle(overlay, leftmost, 5, (255, 0, 0), -1)
    cv2.circle(overlay, rightmost, 5, (0, 0, 255), -1)
    cv2.circle(overlay, min_y_at_max_diff, 5, (0, 255, 255), -1)
    cv2.circle(overlay, max_y_at_max_diff, 5, (255, 0, 255), -1)
    cv2.line(overlay, leftmost, rightmost, (0, 255, 0), 2)
    if min_y_at_max_diff and max_y_at_max_diff:
        cv2.line(overlay, min_y_at_max_diff, max_y_at_max_diff, (255, 255, 0), 2)
    return overlay

def process_images(left_img_path, right_img_path, output_path, intrinsic_matrix1, intrinsic_matrix2, dist_coeffs1, dist_coeffs2, R, T):
    imgL, imgR = load_images(left_img_path, right_img_path)
    jsonL_path = left_img_path.replace('.jpg', '.json')
    jsonR_path = right_img_path.replace('.jpg', '.json')

    with open(jsonL_path, 'r') as f:
        dataL = json.load(f)
    with open(jsonR_path, 'r') as f:
        dataR = json.load(f)

    R1, R2, P1, P2 = stereo_rectification(intrinsic_matrix1, dist_coeffs1, intrinsic_matrix2, dist_coeffs2, imgL.shape[:2], R, T)

    map1L, map2L = cv2.initUndistortRectifyMap(intrinsic_matrix1, dist_coeffs1, R1, P1, imgL.shape[:2], cv2.CV_16SC2)
    map1R, map2R = cv2.initUndistortRectifyMap(intrinsic_matrix2, dist_coeffs2, R2, P2, imgR.shape[:2], cv2.CV_16SC2)

    rectified_imgL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
    rectified_imgR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)

    disparity = compute_disparity(rectified_imgL, rectified_imgR)
    
    disparity_img = compute_disparity(imgL, imgR)
    disparity_normalized = cv2.normalize(disparity_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, os.path.basename(left_img_path).replace('.jpg', '_disparity.jpg')), disparity_normalized)

    focal_length = P1[0, 0]
    baseline = np.linalg.norm(T)
    depth_map = (focal_length * baseline) / (disparity + 1e-5)

    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, os.path.basename(left_img_path).replace('.jpg', '_depth.jpg')), depth_map_normalized)

    pointsL = np.array(dataL['shapes'][0]['points'], dtype=np.int32)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    maskL = create_mask(grayL.shape, pointsL)

    pointsR = np.array(dataR['shapes'][0]['points'], dtype=np.int32)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    maskR = create_mask(grayR.shape, pointsR)

    contours, _ = cv2.findContours(maskL, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    overlay = imgL.copy()
    for contour in contours:
        for point in contour:
            cv2.circle(overlay, tuple(point[0]), 1, (0, 255, 0), -1)

    leftmost, rightmost = find_extreme_points(contours)
    slope, intercept = calculate_line_equation(leftmost, rightmost)
    dx = rightmost[0] - leftmost[0]

    max_height_diff, min_y_at_max_diff, max_y_at_max_diff = calculate_max_height_diff(contours, slope, intercept, dx)
    joint_width_pixels, height_deflection_pixels = calculate_deflection_pixels(leftmost, rightmost, max_height_diff)
    pixel_to_meter_ratio = calculate_pixel_to_meter_ratio(baseline, focal_length, imgL.shape, disparity.shape[1])
    width_deflection_mm, height_deflection_mm = convert_to_mm(joint_width_pixels, height_deflection_pixels, pixel_to_meter_ratio)

    angle = calculate_angle(dx, slope)
    overlay = annotate_image(overlay, leftmost, rightmost, min_y_at_max_diff, max_y_at_max_diff, height_deflection_mm, width_deflection_mm)

    cv2.imwrite(os.path.join(output_path, os.path.basename(left_img_path).replace('.jpg', '_deflection.jpg')), overlay)
    print(f"Deflection analysis results saved to {output_path}")
    print(f"Angle between the lines: {angle:.2f} degrees")

def main():
    
    with open('./stereoParams.json', 'r') as f:
        stereo_params =  json.load(f)
    intrinsic_matrix1, intrinsic_matrix2, dist_coeffs1, dist_coeffs2, R, T = get_camera_params(stereo_params)

    data_path = './data'
    left_images_path = os.path.join(data_path, 'left_images')
    right_images_path = os.path.join(data_path, 'right_images')
    output_path = './output'
    os.makedirs(output_path, exist_ok=True)

    left_images = [f for f in os.listdir(left_images_path) if f.endswith('.jpg')]
    right_images = [f for f in os.listdir(right_images_path) if f.endswith('.jpg')]

    left_images.sort()
    right_images.sort()

    for left_image, right_image in zip(left_images, right_images):
        left_img_path = os.path.join(left_images_path, left_image)
        right_img_path = os.path.join(right_images_path, right_image)
        process_images(left_img_path, right_img_path, output_path, intrinsic_matrix1, intrinsic_matrix2, dist_coeffs1, dist_coeffs2, R, T)

if __name__ == "__main__":
    main()
