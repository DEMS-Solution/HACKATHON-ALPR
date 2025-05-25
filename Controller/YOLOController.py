import os
import cv2
import base64
import numpy as np
import json
import sys
import contextlib

from PIL import Image
from io import BytesIO
from ultralytics import YOLO

from Helpers.Helper import response_api

# Load both YOLO models
vehicle_model = YOLO('/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/DEMS/Config/Yolo/vehicle_detector2.pt') 
plate_model = YOLO('/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/DEMS/Config/Yolo/license_plate_detector2.pt')

base_dir = os.path.dirname(os.path.abspath(__file__))
upload_folder = os.getenv('UPLOAD_FOLDER', os.path.abspath(os.path.join(base_dir, '..', 'Storage', 'Uploads')))
os.makedirs(upload_folder, exist_ok=True)

# Vehicle class mapping (adjust based on your vehicle model)
VEHICLE_CLASSES = {
    'car': ['car', 'sedan', 'suv', 'truck', 'van', 'bus'],
    'motorcycle': ['motorcycle', 'bike', 'scooter', 'motorbike']
}

@contextlib.contextmanager
def delete_debug_yolo():
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
    index = 1
    while True:
        folder_name = f"{index}_img_base64"
        folder_path = os.path.join(upload_folder, folder_name)
        if not os.path.exists(folder_path):
            return folder_name, folder_path
        index += 1

def create_output_folder(filename, is_base64=False):
    if is_base64:
        folder_name, output_dir = get_unique_base64_folder()
        os.makedirs(output_dir, exist_ok=True)
    else:
        base_name = os.path.splitext(filename)[0]
        folder_name = base_name
        output_dir = os.path.join(upload_folder, folder_name)
        os.makedirs(output_dir, exist_ok=True)
    return output_dir, folder_name

def enhance_night_visibility(image):
    """Enhanced preprocessing specifically for night/low-light images"""
    # Convert to different color spaces for better analysis
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    enhanced_images = []
    
    # 1. Extreme CLAHE on LAB L-channel for night images
    clahe_extreme = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4,4))
    lab[:,:,0] = clahe_extreme.apply(lab[:,:,0])
    clahe_result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    enhanced_images.append(("clahe_extreme", clahe_result))
    
    # 2. Histogram equalization on V channel
    hsv_copy = hsv.copy()
    hsv_copy[:,:,2] = cv2.equalizeHist(hsv_copy[:,:,2])
    hsv_eq = cv2.cvtColor(hsv_copy, cv2.COLOR_HSV2BGR)
    enhanced_images.append(("hsv_hist_eq", hsv_eq))
    
    # 3. Adaptive gamma correction (focused on best values for license plates)
    for gamma in [0.4, 0.5, 0.6]:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_img = cv2.LUT(image, table)
        enhanced_images.append((f"gamma_{gamma}", gamma_img))
    
    # 4. Brightness and contrast adjustment (optimized combinations)
    brightness_contrast_configs = [
        (1.8, 80),   # High contrast, very bright
        (2.2, 60),   # Balanced enhancement
        (1.5, 100),  # Very bright
    ]
    
    for alpha, beta in brightness_contrast_configs:
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        enhanced_images.append((f"bright_contrast_{alpha}_{beta}", enhanced))
    
    return enhanced_images

def preprocess_image_optimized(image):
    """Optimized preprocessing with focus on night/challenging conditions"""
    processed_images = []
    
    # Original image
    processed_images.append(("original", image))
    
    # Night-specific enhancements
    night_enhanced = enhance_night_visibility(image)
    processed_images.extend(night_enhanced)
    
    # Additional standard preprocessing
    # CLAHE variations (reduced for efficiency)
    clahe_configs = [(2.0, (8,8)), (4.0, (4,4))]
    for clip_limit, tile_size in clahe_configs:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        processed_images.append((f"clahe_{clip_limit}_{tile_size[0]}x{tile_size[1]}", clahe_img))
    
    # Bilateral filter for noise reduction
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    processed_images.append(("bilateral", bilateral))
    
    return processed_images

def detect_vehicles(image, vehicle_type, debug_info):
    """Stage 1: Detect vehicles in the image"""
    detected_vehicles = []
    
    # Preprocess image for vehicle detection
    processed_images = preprocess_image_optimized(image)
    
    # Vehicle detection parameters
    confidence_thresholds = [0.3, 0.2, 0.1]  # Higher confidence for vehicle detection
    
    debug_info.append("=== STAGE 1: VEHICLE DETECTION ===")
    
    for method_name, proc_img in processed_images[:3]:  # Use top 3 preprocessing methods for efficiency
        for conf_thresh in confidence_thresholds:
            try:
                with delete_debug_yolo():
                    results = vehicle_model(proc_img, conf=conf_thresh, imgsz=640, verbose=False)[0]
                
                if results.boxes is not None and len(results.boxes) > 0:
                    for result in results.boxes:
                        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(result.conf[0].cpu().numpy())
                        class_id = int(result.cls[0].cpu().numpy())
                        
                        # Get class name from model
                        class_name = vehicle_model.names[class_id].lower()
                        
                        # Check if detected vehicle matches the target type
                        is_target_vehicle = False
                        for vehicle_category, class_list in VEHICLE_CLASSES.items():
                            if vehicle_category == vehicle_type and any(vc in class_name for vc in class_list):
                                is_target_vehicle = True
                                break
                        
                        if is_target_vehicle and is_valid_vehicle_detection([x1, y1, x2, y2], image.shape):
                            vehicle_info = {
                                'box': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_name': class_name,
                                'method': method_name
                            }
                            detected_vehicles.append(vehicle_info)
                            debug_info.append(f"Vehicle detected: {class_name}, conf={confidence:.4f}, box=[{x1},{y1},{x2},{y2}]")
            
            except Exception as e:
                debug_info.append(f"Error in vehicle detection with {method_name}: {str(e)}")
                continue
    
    # Remove duplicate vehicles using NMS
    unique_vehicles = non_maximum_suppression_vehicles(detected_vehicles)
    debug_info.append(f"Unique vehicles after NMS: {len(unique_vehicles)}")
    
    return unique_vehicles

def detect_plates_in_vehicles(image, vehicles, vehicle_type, debug_info):
    """Stage 2: Detect license plates within detected vehicles"""
    all_plate_detections = []
    
    debug_info.append("=== STAGE 2: LICENSE PLATE DETECTION ===")
    
    if not vehicles:
        debug_info.append("No vehicles detected, searching entire image for plates")
        # If no vehicles detected, search entire image
        vehicles = [{'box': [0, 0, image.shape[1], image.shape[0]], 'confidence': 1.0, 'class_name': 'unknown'}]
    
    for i, vehicle in enumerate(vehicles):
        vx1, vy1, vx2, vy2 = vehicle['box']
        debug_info.append(f"Searching for plates in vehicle #{i+1}: {vehicle['class_name']}")
        
        # Extract vehicle region with some padding
        padding = 20
        crop_x1 = max(0, vx1 - padding)
        crop_y1 = max(0, vy1 - padding)
        crop_x2 = min(image.shape[1], vx2 + padding)
        crop_y2 = min(image.shape[0], vy2 + padding)
        
        vehicle_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if vehicle_crop.size == 0:
            continue
        
        # Preprocess vehicle crop for plate detection
        processed_crops = preprocess_image_optimized(vehicle_crop)
        
        # Plate detection parameters (more aggressive for cropped regions)
        confidence_thresholds = [0.1, 0.05, 0.03, 0.01]
        iou_thresholds = [0.45, 0.3]
        
        for method_name, proc_crop in processed_crops:
            for conf_thresh in confidence_thresholds:
                for iou_thresh in iou_thresholds:
                    try:
                        with delete_debug_yolo():
                            results = plate_model(proc_crop, conf=conf_thresh, iou=iou_thresh, 
                                                imgsz=640, augment=True, verbose=False)[0]
                        
                        if results.boxes is not None and len(results.boxes) > 0:
                            for result in results.boxes:
                                px1, py1, px2, py2 = result.xyxy[0].cpu().numpy().astype(int)
                                confidence = float(result.conf[0].cpu().numpy())
                                
                                # Convert coordinates back to original image space
                                global_x1 = crop_x1 + px1
                                global_y1 = crop_y1 + py1
                                global_x2 = crop_x1 + px2
                                global_y2 = crop_y1 + py2
                                
                                # Validate plate detection
                                if is_valid_plate_detection_improved([global_x1, global_y1, global_x2, global_y2], 
                                                                   image.shape, vehicle_type):
                                    detection_info = {
                                        'box': [global_x1, global_y1, global_x2, global_y2],
                                        'confidence': confidence,
                                        'method': method_name,
                                        'vehicle_index': i,
                                        'vehicle_confidence': vehicle['confidence']
                                    }
                                    all_plate_detections.append(detection_info)
                                    debug_info.append(f"Plate found in vehicle #{i+1}: conf={confidence:.4f}, box=[{global_x1},{global_y1},{global_x2},{global_y2}]")
                    
                    except Exception as e:
                        debug_info.append(f"Error in plate detection: {str(e)}")
                        continue
    
    return all_plate_detections

def is_valid_vehicle_detection(box, image_shape):
    """Validate vehicle detection"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    img_height, img_width = image_shape[:2]
    img_area = img_height * img_width
    
    # Vehicle should be reasonably sized
    area_ratio = area / img_area
    if area_ratio < 0.01 or area_ratio > 0.8:  # Vehicle should be 1-80% of image
        return False
    
    # Basic size check
    if width < 50 or height < 30:
        return False
    
    return True

def is_valid_plate_detection_improved(box, image_shape, vehicle_type="motorcycle"):
    """Improved validation specifically for motorcycle plates in challenging conditions"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    img_height, img_width = image_shape[:2]
    img_area = img_height * img_width
    
    # More flexible area ratio for challenging conditions
    area_ratio = area / img_area
    if area_ratio < 0.0003 or area_ratio > 0.3:
        return False
    
    # Aspect ratio for motorcycle plates (more flexible)
    aspect_ratio = width / height
    if vehicle_type == "motorcycle":
        # Indonesian motorcycle plates can be more square
        if aspect_ratio < 0.7 or aspect_ratio > 4.0:
            return False
    else:  # car
        if aspect_ratio < 1.5 or aspect_ratio > 6.0:
            return False
    
    # Minimum size check (reduced for distant plates)
    if width < 15 or height < 8:
        return False
    
    return True

def non_maximum_suppression_vehicles(detections, overlap_threshold=0.5):
    """NMS for vehicle detections"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        area1 = (x2 - x1) * (y2 - y1)
        
        should_keep = True
        for kept_detection in keep:
            kx1, ky1, kx2, ky2 = kept_detection['box']
            
            # Calculate intersection
            inter_x1 = max(x1, kx1)
            inter_y1 = max(y1, ky1)
            inter_x2 = min(x2, kx2)
            inter_y2 = min(y2, ky2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area2 = (kx2 - kx1) * (ky2 - ky1)
                
                # Calculate IoU
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > overlap_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            keep.append(detection)
    
    return keep

def non_maximum_suppression(detections, overlap_threshold=0.3):
    """Custom NMS for better duplicate removal"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence, but also consider vehicle confidence
    detections.sort(key=lambda x: (x['confidence'] + x.get('vehicle_confidence', 0)) / 2, reverse=True)
    
    keep = []
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        area1 = (x2 - x1) * (y2 - y1)
        
        should_keep = True
        for kept_detection in keep:
            kx1, ky1, kx2, ky2 = kept_detection['box']
            
            # Calculate intersection
            inter_x1 = max(x1, kx1)
            inter_y1 = max(y1, ky1)
            inter_x2 = min(x2, kx2)
            inter_y2 = min(y2, ky2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area2 = (kx2 - kx1) * (ky2 - ky1)
                
                # Calculate IoU
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > overlap_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            keep.append(detection)
    
    return keep

def detect_plate(image_input, vehicle_type):
    is_base64 = False
    save_visualization = True

    if vehicle_type not in ["car", "motorcycle"]:
        return response_api(400, 'Error', 'Invalid vehicle type', 'Only car or motorcycle are supported.')

    # Load image
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
    
    # Stage 1: Detect vehicles
    detected_vehicles = detect_vehicles(image, vehicle_type, debug_info)
    
    # Stage 2: Detect plates within vehicles
    all_plate_detections = detect_plates_in_vehicles(image, detected_vehicles, vehicle_type, debug_info)
    
    # Save debug info
    debug_path = os.path.join(output_dir, "debug_log.txt")
    with open(debug_path, 'w') as f:
        f.write(f"=== TWO-STAGE DETECTION DEBUG LOG ===\n")
        f.write(f"Total vehicles detected: {len(detected_vehicles)}\n")
        f.write(f"Total plate detections: {len(all_plate_detections)}\n")
        f.write("\n".join(debug_info))
    
    if not all_plate_detections:
        return response_api(400, 'Error', 'No plate detected', 
                          f'No license plate was detected. Vehicles found: {len(detected_vehicles)}. Check debug log at: {debug_path}')
    
    # Apply NMS to plate detections
    unique_plate_detections = non_maximum_suppression(all_plate_detections, overlap_threshold=0.3)
    
    # Update debug log with final results
    with open(debug_path, 'a') as f:
        f.write(f"\n=== FINAL RESULTS ===\n")
        f.write(f"Unique plate detections after NMS: {len(unique_plate_detections)}\n")
        for i, det in enumerate(unique_plate_detections[:3]):
            f.write(f"#{i+1}: conf={det['confidence']:.4f}, method={det['method']}, vehicle_idx={det.get('vehicle_index', 'N/A')}\n")
    
    # Select best detection
    best_detection = unique_plate_detections[0]
    x1, y1, x2, y2 = best_detection['box']
    h, w = image.shape[:2]

    # Smart cropping with context
    detection_width = x2 - x1
    detection_height = y2 - y1
    
    # Adaptive margins based on detection size and image size
    margin_x = max(20, min(int(detection_width * 0.3), w // 10))
    margin_y = max(15, min(int(detection_height * 0.3), h // 10))
    
    crop_x1 = max(x1 - margin_x, 0)
    crop_y1 = max(y1 - margin_y, 0)
    crop_x2 = min(x2 + margin_x, w - 1)
    crop_y2 = min(y2 + margin_y, h - 1)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Enhanced crop post-processing
    if crop.size > 0:
        # Apply additional enhancement to the crop
        crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        crop_lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(crop_lab[:,:,0])
        crop = cv2.cvtColor(crop_lab, cv2.COLOR_LAB2BGR)
    
    detect_plate_color = 'UNKNOWN'

    if save_visualization:
        # Create visualizations
        vis_crop = crop.copy()
        vis_full = image.copy()
        
        # Draw vehicle bounding boxes
        for i, vehicle in enumerate(detected_vehicles):
            vx1, vy1, vx2, vy2 = vehicle['box']
            cv2.rectangle(vis_full, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)  # Blue for vehicles
            cv2.putText(vis_full, f"Vehicle {i+1}", (vx1, vy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw license plate bounding box
        cv2.rectangle(vis_full, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for license plate
        cv2.putText(vis_full, "License Plate", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw bounding box on crop
        if crop.size > 0:
            box_top_left = (x1 - crop_x1, y1 - crop_y1)
            box_bottom_right = (x2 - crop_x1, y2 - crop_y1)
            cv2.rectangle(vis_crop, box_top_left, box_bottom_right, (0, 255, 0), 1)

        # Generate filenames
        base_name = os.path.splitext(filename)[0]
        crop_filename = f"{base_name}_detected_crop.jpg"
        full_filename = f"{base_name}_detected_full.jpg"
        original_filename = f"{base_name}_original.jpg"
        
        # Save files
        crop_path = os.path.join(output_dir, crop_filename)
        full_path = os.path.join(output_dir, full_filename)
        original_path = os.path.join(output_dir, original_filename)
        
        if crop.size > 0:
            cv2.imwrite(crop_path, vis_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(full_path, vis_full, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(original_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return response_api(
        200, 'Success', 'Plate detected successfully',
        {
            'plate_type': vehicle_type,
            'plate_color': detect_plate_color,
            'plate_number': 'UNKNOWN',
            'plate_confidence': best_detection['confidence'],
            'output_folder': output_dir,
        }
    )

if __name__ == "__main__":
    # TEST DENGAN PATH IMAGE LANGSUNG
    # result = detect_plate(
    #     '/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/Testing Apps/Storage/test-gambar/gambar14.jpg',
    #     "motorcycle"
    # )
    # print(json.dumps(result, indent=4))
    
    # TEST DENGAN BASE64 INPUT
    with open('/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/Testing Apps/Storage/test-gambar/gambar28.jpeg', "rb") as f:
        img_bytes = f.read()
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        b64_input = f"data:image/jpeg;base64,{b64_str}"
    
    with delete_debug_yolo():
        result_b64 = detect_plate(b64_input, "motorcycle")

    result_b64.pop('debug', None)
    print(json.dumps(result_b64, indent=4))