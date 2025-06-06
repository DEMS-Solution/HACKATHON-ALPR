import os
import cv2
import base64
import numpy as np
import json
import sys
import contextlib
import concurrent.futures

from typing import List, Tuple, Dict, Any
from functools import lru_cache
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from Controller.Helpers.Helper import response_api

# Load YOLO models - separated for better specialization
car_model = YOLO('/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/DEMS/HACKATHON-ALPR/Config/Yolo/car.pt')
motorcycle_model = YOLO('/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/DEMS/HACKATHON-ALPR/Config/Yolo/motor.pt') 
plate_model = YOLO('/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/DEMS/HACKATHON-ALPR/Config/Yolo/plate.pt')

base_dir = os.path.dirname(os.path.abspath(__file__))
upload_folder = os.getenv('UPLOAD_FOLDER', os.path.abspath(os.path.join(base_dir, '..', 'Storage', 'Uploads')))
os.makedirs(upload_folder, exist_ok=True)

# Optimized preprocessing parameters
PREPROCESSING_CONFIGS = {
    'car': {
        'gamma_values': [0.6, 0.8, 1.2],
        'clahe_configs': [(3.0, (8,8)), (5.0, (4,4))],
        'brightness_contrast': [(1.3, 40), (1.6, 60)],
        'confidence_thresholds': [0.25, 0.15, 0.1]
    },
    'motorcycle': {
        'gamma_values': [0.4, 0.5, 0.7],
        'clahe_configs': [(4.0, (4,4)), (6.0, (2,2))],
        'brightness_contrast': [(1.8, 80), (2.2, 60)],
        'confidence_thresholds': [0.2, 0.1, 0.05]
    }
}

@contextlib.contextmanager
def delete_debug_yolo():
    """Suppress YOLO output for cleaner processing"""
    with open(os.devnull, 'w') as devnull:
        old_stdout_fileno = os.dup(sys.stdout.fileno())
        old_stderr_fileno = os.dup(sys.stderr.fileno())
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            yield
        finally:
            os.dup2(old_stdout_fileno, sys.stdout.fileno())
            os.dup2(old_stderr_fileno, sys.stderr.fileno())

def get_unique_base64_folder():
    """Generate unique folder for base64 images"""
    index = 1
    while True:
        folder_name = f"{index}_img_base64"
        folder_path = os.path.join(upload_folder, folder_name)
        if not os.path.exists(folder_path):
            return folder_name, folder_path
        index += 1

def create_output_folder(filename: str, is_base64: bool = False) -> Tuple[str, str]:
    """Create output folder for results"""
    if is_base64:
        folder_name, output_dir = get_unique_base64_folder()
        os.makedirs(output_dir, exist_ok=True)
    else:
        base_name = os.path.splitext(filename)[0]
        folder_name = base_name
        output_dir = os.path.join(upload_folder, folder_name)
        os.makedirs(output_dir, exist_ok=True)
    return output_dir, folder_name

@lru_cache(maxsize=32)
def get_gamma_table(gamma: float) -> np.ndarray:
    """Cached gamma correction table for performance"""
    inv_gamma = 1.0 / gamma
    return np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

def apply_advanced_clahe(image: np.ndarray, clip_limit: float, tile_size: Tuple[int, int]) -> np.ndarray:
    """Apply CLAHE with optimized parameters"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_multi_scale_enhancement(image: np.ndarray) -> np.ndarray:
    """Multi-scale enhancement for better plate visibility"""
    # Gaussian pyramid for multi-scale processing
    pyramid = [image]
    temp = image.copy()
    for _ in range(2):
        temp = cv2.pyrDown(temp)
        pyramid.append(temp)
    
    # Process each scale
    enhanced_pyramid = []
    for img in pyramid:
        # Enhanced CLAHE
        enhanced = apply_advanced_clahe(img, 4.0, (4, 4))
        # Sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        enhanced_pyramid.append(enhanced)
    
    # Reconstruct from pyramid
    result = enhanced_pyramid[0]
    for i in range(1, len(enhanced_pyramid)):
        upsampled = enhanced_pyramid[i]
        for _ in range(i):
            upsampled = cv2.pyrUp(upsampled)
        # Resize to match original if needed
        if upsampled.shape[:2] != result.shape[:2]:
            upsampled = cv2.resize(upsampled, (result.shape[1], result.shape[0]))
        result = cv2.addWeighted(result, 0.7, upsampled, 0.3, 0)
    
    return result

def enhance_for_plate_detection(image: np.ndarray, vehicle_type: str) -> List[Tuple[str, np.ndarray]]:
    """Enhanced preprocessing specifically optimized for license plate detection"""
    config = PREPROCESSING_CONFIGS[vehicle_type]
    processed_images = []
    
    # Original image
    processed_images.append(("original", image))
    
    # Multi-scale enhancement
    multi_scale = apply_multi_scale_enhancement(image)
    processed_images.append(("multi_scale", multi_scale))
    
    # Optimized gamma correction
    for gamma in config['gamma_values']:
        table = get_gamma_table(gamma)
        gamma_img = cv2.LUT(image, table)
        processed_images.append((f"gamma_{gamma}", gamma_img))
    
    # Advanced CLAHE variations
    for clip_limit, tile_size in config['clahe_configs']:
        clahe_img = apply_advanced_clahe(image, clip_limit, tile_size)
        processed_images.append((f"clahe_{clip_limit}_{tile_size[0]}x{tile_size[1]}", clahe_img))
    
    # Brightness and contrast adjustment
    for alpha, beta in config['brightness_contrast']:
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        processed_images.append((f"bright_contrast_{alpha}_{beta}", enhanced))
    
    # Edge enhancement for plate boundaries
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edge_enhanced = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
    processed_images.append(("edge_enhanced", edge_enhanced))
    
    # Histogram equalization on HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    hsv_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    processed_images.append(("hsv_equalized", hsv_eq))
    
    return processed_images

def detect_vehicles_parallel(image: np.ndarray, vehicle_type: str, debug_info: List[str]) -> List[Dict[str, Any]]:
    """Optimized vehicle detection with parallel processing"""
    detected_vehicles = []
    
    # Select appropriate model
    model = car_model if vehicle_type == "car" else motorcycle_model
    config = PREPROCESSING_CONFIGS[vehicle_type]
    
    debug_info.append(f"=== STAGE 1: {vehicle_type.upper()} DETECTION ===")
    
    # Get preprocessed images
    processed_images = enhance_for_plate_detection(image, vehicle_type)
    
    # Use only top performing preprocessing methods for speed
    top_methods = processed_images[:4]  # Limit to 4 best methods
    
    def detect_in_image(args):
        method_name, proc_img, conf_thresh = args
        detections = []
        try:
            results = model(proc_img, conf=conf_thresh, imgsz=640, verbose=False)[0]
            
            if results.boxes is not None and len(results.boxes) > 0:
                for result in results.boxes:
                    x1, y1, x2, y2 = result.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(result.conf[0].cpu().numpy())
                    class_id = int(result.cls[0].cpu().numpy())
                    class_name = model.names[class_id].lower()
                    
                    if is_valid_vehicle_detection([x1, y1, x2, y2], image.shape, vehicle_type):
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_name': class_name,
                            'method': method_name
                        })
        except Exception as e:
            debug_info.append(f"Error in vehicle detection: {str(e)}")
        
        return detections
    
    # Parallel processing
    tasks = []
    for method_name, proc_img in top_methods:
        for conf_thresh in config['confidence_thresholds']:
            tasks.append((method_name, proc_img, conf_thresh))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(detect_in_image, tasks)
        for result in results:
            detected_vehicles.extend(result)
    
    # Remove duplicates using optimized NMS
    unique_vehicles = non_maximum_suppression_optimized(detected_vehicles, overlap_threshold=0.5)
    debug_info.append(f"Unique vehicles after NMS: {len(unique_vehicles)}")
    
    return unique_vehicles

def detect_plates_in_regions(image: np.ndarray, regions: List[Dict], vehicle_type: str, debug_info: List[str]) -> List[Dict[str, Any]]:
    """Optimized plate detection within vehicle regions"""
    all_plate_detections = []
    config = PREPROCESSING_CONFIGS[vehicle_type]
    
    debug_info.append("=== STAGE 2: LICENSE PLATE DETECTION ===")
    
    if not regions:
        debug_info.append("No vehicles detected, searching entire image")
        regions = [{'box': [0, 0, image.shape[1], image.shape[0]], 'confidence': 1.0, 'class_name': 'full_image'}]
    
    def detect_plates_in_region(args):
        region_idx, region = args
        region_detections = []
        
        vx1, vy1, vx2, vy2 = region['box']
        
        # Smart cropping with adaptive padding
        padding_x = max(10, int((vx2 - vx1) * 0.1))
        padding_y = max(10, int((vy2 - vy1) * 0.1))
        
        crop_x1 = max(0, vx1 - padding_x)
        crop_y1 = max(0, vy1 - padding_y)
        crop_x2 = min(image.shape[1], vx2 + padding_x)
        crop_y2 = min(image.shape[0], vy2 + padding_y)
        
        vehicle_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if vehicle_crop.size == 0:
            return region_detections
        
        # Enhanced preprocessing for the crop
        processed_crops = enhance_for_plate_detection(vehicle_crop, vehicle_type)
        
        # Use top 3 preprocessing methods for speed
        for method_name, proc_crop in processed_crops[:3]:
            # More aggressive confidence thresholds for cropped regions
            for conf_thresh in [0.05, 0.03, 0.01]:
                try:
                    results = plate_model(proc_crop, conf=conf_thresh, iou=0.3, 
                                            imgsz=640, augment=True, verbose=False)[0]
                    
                    if results.boxes is not None and len(results.boxes) > 0:
                        for result in results.boxes:
                            px1, py1, px2, py2 = result.xyxy[0].cpu().numpy().astype(int)
                            confidence = float(result.conf[0].cpu().numpy())
                            
                            # Convert to global coordinates
                            global_x1 = crop_x1 + px1
                            global_y1 = crop_y1 + py1
                            global_x2 = crop_x1 + px2
                            global_y2 = crop_y1 + py2
                            
                            if is_valid_plate_detection_optimized([global_x1, global_y1, global_x2, global_y2], 
                                                                image.shape, vehicle_type):
                                region_detections.append({
                                    'box': [global_x1, global_y1, global_x2, global_y2],
                                    'confidence': confidence,
                                    'method': method_name,
                                    'vehicle_index': region_idx,
                                    'vehicle_confidence': region['confidence']
                                })
                
                except Exception as e:
                    debug_info.append(f"Error in plate detection: {str(e)}")
                    continue
        
        return region_detections
    
    # Parallel processing of regions
    tasks = [(i, region) for i, region in enumerate(regions)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(detect_plates_in_region, tasks)
        for result in results:
            all_plate_detections.extend(result)
    
    return all_plate_detections

def is_valid_vehicle_detection(box: List[int], image_shape: Tuple[int, int, int], vehicle_type: str) -> bool:
    """Enhanced vehicle validation with type-specific criteria"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    img_height, img_width = image_shape[:2]
    img_area = img_height * img_width
    area_ratio = area / img_area
    
    if vehicle_type == "car":
        # Cars are generally larger
        if area_ratio < 0.02 or area_ratio > 0.85:
            return False
        if width < 80 or height < 40:
            return False
        # Car aspect ratio check
        aspect_ratio = width / height
        if aspect_ratio < 0.8 or aspect_ratio > 3.5:
            return False
    else:  # motorcycle
        # Motorcycles can be smaller
        if area_ratio < 0.005 or area_ratio > 0.7:
            return False
        if width < 30 or height < 40:
            return False
        # Motorcycle aspect ratio check
        aspect_ratio = width / height
        if aspect_ratio < 0.3 or aspect_ratio > 2.5:
            return False
    
    return True

def is_valid_plate_detection_optimized(box: List[int], image_shape: Tuple[int, int, int], vehicle_type: str) -> bool:
    """Optimized plate validation with better criteria"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    img_height, img_width = image_shape[:2]
    img_area = img_height * img_width
    area_ratio = area / img_area
    
    # More flexible area constraints
    if area_ratio < 0.0002 or area_ratio > 0.25:
        return False
    
    # Type-specific aspect ratio validation
    aspect_ratio = width / height
    if vehicle_type == "motorcycle":
        # Indonesian motorcycle plates: more square format
        if aspect_ratio < 0.6 or aspect_ratio > 4.5:
            return False
        # Minimum size for motorcycles (can be smaller/distant)
        if width < 12 or height < 6:
            return False
    else:  # car
        # Car plates: typically rectangular  
        if aspect_ratio < 1.2 or aspect_ratio > 6.5:
            return False
        # Minimum size for cars
        if width < 20 or height < 8:
            return False
    
    # Check if plate is not too thin or too small
    if width < 8 or height < 4:
        return False
    
    return True

def non_maximum_suppression_optimized(detections: List[Dict], overlap_threshold: float = 0.4) -> List[Dict]:
    """Optimized NMS with better scoring"""
    if len(detections) == 0:
        return []
    
    # Enhanced scoring: combine confidence with detection quality metrics
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height
        
        # Quality score based on size and aspect ratio
        size_score = min(1.0, (width * height) / 10000)  # Normalize by typical plate size
        aspect_score = 1.0 if 1.5 <= aspect_ratio <= 4.0 else 0.7  # Prefer typical plate ratios
        
        detection['quality_score'] = detection['confidence'] * size_score * aspect_score
    
    # Sort by enhanced quality score
    detections.sort(key=lambda x: x['quality_score'], reverse=True)
    
    keep = []
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        area1 = (x2 - x1) * (y2 - y1)
        
        should_keep = True
        for kept_detection in keep:
            kx1, ky1, kx2, ky2 = kept_detection['box']
            
            # Calculate IoU
            inter_x1 = max(x1, kx1)
            inter_y1 = max(y1, ky1)
            inter_x2 = min(x2, kx2)
            inter_y2 = min(y2, ky2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area2 = (kx2 - kx1) * (ky2 - ky1)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > overlap_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            keep.append(detection)
    
    return keep

def apply_smart_crop_enhancement(crop: np.ndarray, vehicle_type: str) -> np.ndarray:
    """Apply final enhancement to the cropped plate region"""
    if crop.size == 0:
        return crop
    
    # Multi-stage enhancement
    enhanced = crop.copy()
    
    # Stage 1: Noise reduction while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Stage 2: Adaptive enhancement based on vehicle type
    if vehicle_type == "motorcycle":
        # More aggressive enhancement for motorcycle plates (often smaller/distant)
        enhanced = apply_advanced_clahe(enhanced, 6.0, (2, 2))
        # Additional gamma correction
        table = get_gamma_table(0.6)
        enhanced = cv2.LUT(enhanced, table)
    else:  # car
        # Moderate enhancement for car plates
        enhanced = apply_advanced_clahe(enhanced, 4.0, (4, 4))
        table = get_gamma_table(0.8)
        enhanced = cv2.LUT(enhanced, table)
    
    # Stage 3: Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Stage 4: Final contrast adjustment
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
    
    return enhanced

def detect_plate(image_input, vehicle_type: str):
    """Main detection function with optimized pipeline"""
    is_base64 = False
    save_visualization = True

    if vehicle_type not in ["car", "motorcycle"]:
        return response_api(400, 'Error', 'Invalid vehicle type', 'Only car or motorcycle are supported.')

    # Load and validate image
    if isinstance(image_input, str):
        if os.path.isfile(image_input):
            image = cv2.imread(image_input)
            filename = os.path.basename(image_input)
        else:
            try:
                header_removed = image_input.split(',')[-1]
                img_data = base64.b64decode(header_removed)
                image_np = np.array(Image.open(BytesIO(img_data)).convert('RGB'))
                image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                is_base64 = True
                filename = "img64.jpeg"
            except Exception as e:
                return response_api(400, 'Error', 'Invalid image input', str(e))
    else:
        return response_api(400, 'Error', 'Invalid image input type', 'Input must be a file path or base64 string.')

    if image is None:
        return response_api(400, 'Error', 'Image not found or invalid', 'Could not read the image.')

    # Create output folder
    output_dir, folder_name = create_output_folder(filename, is_base64)
    
    debug_info = []
    debug_info.append(f"Image shape: {image.shape}")
    debug_info.append(f"Target vehicle type: {vehicle_type}")
    debug_info.append(f"Using optimized detection pipeline")
    
    # Stage 1: Detect vehicles with parallel processing
    detected_vehicles = detect_vehicles_parallel(image, vehicle_type, debug_info)
    
    # Stage 2: Detect plates within vehicle regions
    all_plate_detections = detect_plates_in_regions(image, detected_vehicles, vehicle_type, debug_info)
    
    # Save debug information
    debug_path = os.path.join(output_dir, "debug_log.txt")
    with open(debug_path, 'w') as f:
        f.write(f"=== OPTIMIZED DETECTION DEBUG LOG ===\n")
        f.write(f"Total vehicles detected: {len(detected_vehicles)}\n")
        f.write(f"Total plate detections: {len(all_plate_detections)}\n")
        f.write("\n".join(debug_info))
    
    if not all_plate_detections:
        return response_api(400, 'Error', 'No plate detected', 
                          f'No license plate detected. Vehicles found: {len(detected_vehicles)}. Check debug: {debug_path}')
    
    # Apply optimized NMS
    unique_plate_detections = non_maximum_suppression_optimized(all_plate_detections, overlap_threshold=0.3)
    
    # Update debug with final results
    with open(debug_path, 'a') as f:
        f.write(f"\n=== FINAL RESULTS ===\n")
        f.write(f"Unique plates after optimized NMS: {len(unique_plate_detections)}\n")
        for i, det in enumerate(unique_plate_detections[:3]):
            f.write(f"#{i+1}: conf={det['confidence']:.4f}, quality={det.get('quality_score', 0):.4f}\n")
    
    # Select best detection
    best_detection = unique_plate_detections[0]
    x1, y1, x2, y2 = best_detection['box']
    h, w = image.shape[:2]

    # Smart adaptive cropping
    detection_width = x2 - x1
    detection_height = y2 - y1
    
    # Adaptive margins based on vehicle type and detection size
    if vehicle_type == "motorcycle":
        margin_factor = 0.4  # Larger margins for motorcycle plates
    else:
        margin_factor = 0.25  # Smaller margins for car plates
    
    margin_x = max(15, min(int(detection_width * margin_factor), w // 8))
    margin_y = max(10, min(int(detection_height * margin_factor), h // 8))
    
    crop_x1 = max(x1 - margin_x, 0)
    crop_y1 = max(y1 - margin_y, 0)
    crop_x2 = min(x2 + margin_x, w - 1)
    crop_y2 = min(y2 + margin_y, h - 1)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Apply smart enhancement to crop
    if crop.size > 0:
        crop = apply_smart_crop_enhancement(crop, vehicle_type)

    if save_visualization:
        # Create clean visualizations without boundary boxes or text
        vis_full = image.copy()
        
        # Only save clean images without annotations
        base_name = os.path.splitext(filename)[0]
        crop_filename = f"{base_name}_detected_crop.jpg"
        full_filename = f"{base_name}_detected_full.jpg"
        original_filename = f"{base_name}_original.jpg"
        
        # Save files
        crop_path = os.path.join(output_dir, crop_filename)
        full_path = os.path.join(output_dir, full_filename)
        original_path = os.path.join(output_dir, original_filename)
        
        if crop.size > 0:
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.rectangle(vis_full, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
        cv2.imwrite(full_path, vis_full, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(original_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return crop_path, full_path, original_path, vehicle_type