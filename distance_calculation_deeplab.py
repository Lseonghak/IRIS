import torch
from torchvision import models
from torchvision.transforms import functional as F
import numpy as np
import cv2
import os
import time

model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
checkpoint = torch.load('./checkpoints/deeplabv3_finetuned.pth')
model.load_state_dict(checkpoint, strict=False)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

video_path = './paul.MP4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

output_path = './output_deeplab_paul.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

total_frames = 0
start_time = time.time() 

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = F.to_tensor(frame)
    frame = F.normalize(frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return frame.unsqueeze(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    mask = (output_predictions == 1).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        coords = contour.squeeze(1)

        for coord in coords:
            cv2.circle(frame, tuple(coord), 3, (0, 255, 0), -1) 

        leftmost = tuple(coords[coords[:, 0].argmin()])
        rightmost = tuple(coords[coords[:, 0].argmax()])
        topmost = tuple(coords[coords[:, 1].argmin()])
        bottommost = tuple(coords[coords[:, 1].argmax()])

        left_right_length = np.abs(rightmost[0] - leftmost[0])
        top_bottom_length = np.abs(bottommost[1] - topmost[1])

        cv2.putText(frame, f"Left-Right Length: {left_right_length}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Top-Bottom Length: {top_bottom_length}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(frame)
    total_frames += 1

cap.release()
out.release()
cv2.destroyAllWindows()

end_time = time.time() 
total_time = end_time - start_time  
fps = total_frames / total_time  

print(f"Video processing completed. Output saved to: {output_path}")
#print(f"Total frames processed: {total_frames}")
#print(f"Total processing time: {total_time:.2f} seconds")
#print(f"Calculated FPS: {fps:.2f}")
