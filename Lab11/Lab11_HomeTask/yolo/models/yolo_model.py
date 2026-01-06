from ultralytics import YOLO
import os

# Load YOLO11 models for different tasks
# detection_model = YOLO('yolo11n.pt')           # Object Detection
# segmentation_model = YOLO('yolo11n-seg.pt')    # Instance Segmentation
# classification_model = YOLO('yolo11n-cls.pt')  # Classification
# pose_model = YOLO('yolo11n-pose.pt')           # Pose Estimation
# obb_model = YOLO('yolo11n-obb.pt')             # Oriented Bounding Boxes

detection_model = YOLO(os.path.join('models', 'yolo11n.pt'))
segmentation_model = YOLO(os.path.join('models', 'yolo11n-seg.pt'))
classification_model = YOLO(os.path.join('models', 'yolo11n-cls.pt'))
pose_model = YOLO(os.path.join('models', 'yolo11n-pose.pt'))
obb_model = YOLO(os.path.join('models', 'yolo11n-obb.pt'))