from models.yolo_model import (
    detection_model, 
    segmentation_model, 
    classification_model, 
    pose_model, 
    obb_model
)
from PIL import Image
import io


def process_detection(file_storage):
    """Object Detection - Find objects and their locations"""
    image_bytes = file_storage.read()
    img = Image.open(io.BytesIO(image_bytes))

    results = detection_model(img)
    
    detections = []
    for result in results:
        class_names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
    
    return {"task": "detection", "detections": detections}


def process_segmentation(file_storage):
    """Instance Segmentation - Get pixel-level masks"""
    image_bytes = file_storage.read()
    img = Image.open(io.BytesIO(image_bytes))

    results = segmentation_model(img)
    
    segments = []
    for result in results:
        class_names = result.names
        
        if result.masks is not None:
            for i, mask in enumerate(result.masks):
                box = result.boxes[i]
                class_id = int(box.cls[0])
                
                segments.append({
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                    "mask": mask.xy[0].tolist()  # Polygon points
                })
    
    return {"task": "segmentation", "segments": segments}


def process_classification(file_storage):
    """Classification - Identify main object in image"""
    image_bytes = file_storage.read()
    img = Image.open(io.BytesIO(image_bytes))

    results = classification_model(img)
    
    classifications = []
    for result in results:
        # Get top 5 predictions
        top5_indices = result.probs.top5
        top5_conf = result.probs.top5conf.tolist()
        class_names = result.names
        
        for idx, conf in zip(top5_indices, top5_conf):
            classifications.append({
                "class_id": int(idx),
                "class_name": class_names[int(idx)],
                "confidence": float(conf)
            })
    
    return {"task": "classification", "predictions": classifications}


def process_pose(file_storage):
    """Pose Estimation - Detect human body keypoints"""
    image_bytes = file_storage.read()
    img = Image.open(io.BytesIO(image_bytes))

    results = pose_model(img)
    
    poses = []
    for result in results:
        if result.keypoints is not None:
            for i, keypoints in enumerate(result.keypoints):
                box = result.boxes[i]
                
                # Keypoint names (COCO format)
                keypoint_names = [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle"
                ]
                
                kpts = []
                for j, (x, y, conf) in enumerate(keypoints.xy[0]):
                    if j < len(keypoint_names):
                        kpts.append({
                            "name": keypoint_names[j],
                            "x": float(x),
                            "y": float(y),
                            "confidence": float(conf)
                        })
                
                poses.append({
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "keypoints": kpts
                })
    
    return {"task": "pose_estimation", "poses": poses}


def process_obb(file_storage):
    """Oriented Bounding Boxes - Detect rotated objects"""
    image_bytes = file_storage.read()
    img = Image.open(io.BytesIO(image_bytes))

    results = obb_model(img)
    
    detections = []
    for result in results:
        class_names = result.names
        
        if result.obb is not None:
            for i, obb in enumerate(result.obb):
                detections.append({
                    "class_id": int(obb.cls[0]),
                    "class_name": class_names[int(obb.cls[0])],
                    "confidence": float(obb.conf[0]),
                    "rotated_bbox": obb.xyxyxyxy[0].tolist()  # 4 corner points
                })
    
    return {"task": "oriented_detection", "detections": detections}