import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from ultralytics.hub import hub
from ultralytics import YOLO
import cv2
import os


# 모델 초기화
model_path = './best.pt'
model = YOLO(model_path)

# 동영상 파일 경로
video_path = './paul.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# 출력 동영상 설정
output_path = './output_paul.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지를 모델에 전달하고 결과를 받음
    results = model(frame)

    # 각 결과에 대한 마스크를 저장하고 시각화
    for result in results:
        if result.masks is not None:  # 마스크가 None이 아닐 때만 처리
            for mask in result.masks:
                if mask is not None and hasattr(mask, 'xy') and mask.xy is not None:
                    coords = mask.xy[0]
                    # 각 좌표에 점 그리기
                    for coord in coords:
                        cv2.circle(frame, (int(coord[0]), int(coord[1])), 3, (0, 255, 0), -1)  # 녹색 점

                    # 극단점 계산
                    leftmost = min(coords, key=lambda x: x[0])
                    rightmost = max(coords, key=lambda x: x[0])
                    topmost = min(coords, key=lambda x: x[1])
                    bottommost = max(coords, key=lambda x: x[1])

                    # 길이 계산
                    left_right_length = np.abs(rightmost[0] - leftmost[0])
                    top_bottom_length = np.abs(bottommost[1] - topmost[1])

                    # 텍스트 추가
                    cv2.putText(frame, f"Left-Right Length: {left_right_length}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Top-Bottom Length: {top_bottom_length}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 프레임을 동영상 파일에 쓰기
    out.write(frame)

# 모든 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed. Output saved to:", output_path)