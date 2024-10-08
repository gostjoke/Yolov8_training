from ultralytics import YOLO
import os

# Initialize YOLO with the model file
model = YOLO("yolov8n.pt")

# Define source video and output video paths
source_video = 'demo.mp4'
output_folder = "video_pre"  # Get the same folder as the source video
output_name = 'demo_pred.mp4'  # Define the desired output video name

# Perform prediction on the video file using CUDA and save to the specified path
results = model.predict(source=source_video, save=True, project=output_folder, name=output_name, device='cuda:0')

# You can optionally print the results or do further processing
print(results)
