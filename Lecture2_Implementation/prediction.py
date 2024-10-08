from ultralytics import YOLO
import os

# Initialize YOLO with the Model Name
model = YOLO("yolov8n.pt")

# Perform prediction and save results in the current directory
current_directory = os.getcwd()  # Get the current working directory

# Predict and save the results to the current directory

# model.predict(source='image1.jpg', save=True, conf=0.5, save_txt=True)
model.predict(source='people.jpg', save=True, conf=0.5, save_txt=True, project=current_directory, name='results')

# Export the model in ONNX format and save it in the current directory
model.export(format="onnx", path=os.path.join(current_directory, 'yolov8n.onnx'))
