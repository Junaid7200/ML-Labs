from flask import Blueprint, request, jsonify, render_template
from services.yolo_service import (
    process_detection,
    process_segmentation,
    process_classification,
    process_pose,
    process_obb
)

yolo_bp = Blueprint('yolo', __name__)

@yolo_bp.route('/')
def home():
    """Render the main HTML page"""
    return render_template('main.html')

@yolo_bp.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    result = process_detection(file)
    return jsonify(result)

@yolo_bp.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    result = process_segmentation(file)
    return jsonify(result)

@yolo_bp.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    result = process_classification(file)
    return jsonify(result)

@yolo_bp.route('/pose', methods=['POST'])
def pose():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    result = process_pose(file)
    return jsonify(result)

@yolo_bp.route('/obb', methods=['POST'])
def obb():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    result = process_obb(file)
    return jsonify(result)