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

# Load YOLO model
yolo_model = YOLO('/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/Testing Apps/Config/Yolo/license_plate_detector.pt')

base_dir = os.path.dirname(os.path.abspath(__file__))
upload_folder = os.getenv('UPLOAD_FOLDER', os.path.abspath(os.path.join(base_dir, '..', 'Storage', 'Uploads')))
os.makedirs(upload_folder, exist_ok=True)

@contextlib.contextmanager
def delete_debug_yolo():
    with open(os.devnull, 'w') as devnull:
        # Save actual stdout/stderr file descriptors
        old_stdout_fileno = os.dup(sys.stdout.fileno())
        old_stderr_fileno = os.dup(sys.stderr.fileno())

        # Redirect stdout and stderr to devnull
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())

        try:
            yield
        finally:
            # Restore original stdout and stderr
            os.dup2(old_stdout_fileno, sys.stdout.fileno())
            os.dup2(old_stderr_fileno, sys.stderr.fileno())

def get_unique_base64_folder():
    """
    Get unique folder name for base64 inputs
    """
    index = 1
    while True:
        folder_name = f"{index}_img_base64"
        folder_path = os.path.join(upload_folder, folder_name)
        if not os.path.exists(folder_path):
            return folder_name, folder_path
        index += 1

def create_output_folder(filename, is_base64=False):
    """
    Create organized folder structure for outputs
    """
    if is_base64:
        folder_name, output_dir = get_unique_base64_folder()
        os.makedirs(output_dir, exist_ok=True)
    else:
        base_name = os.path.splitext(filename)[0]
        folder_name = base_name
        output_dir = os.path.join(upload_folder, folder_name)
        os.makedirs(output_dir, exist_ok=True)
    
    return output_dir, folder_name

def preprocess_image_aggressive(image):
    """
    More aggressive preprocessing for challenging detection cases
    """
    processed_images = []
    
    # Original image
    processed_images.append(("original", image))
    
    # Multiple contrast/brightness variations
    contrast_brightness_configs = [
        (1.3, 40),   # High contrast, bright
        (1.5, 50),   # Very high contrast
        (0.8, 60),   # Low contrast, very bright
        (1.2, -20),  # High contrast, darker
        (2.0, 30),   # Extreme contrast
    ]
    
    for alpha, beta in contrast_brightness_configs:
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        processed_images.append((f"contrast_{alpha}_bright_{beta}", enhanced))
    
    # Histogram equalization variations
    # RGB channels separately
    for i, channel_name in enumerate(['B', 'G', 'R']):
        img_copy = image.copy()
        img_copy[:,:,i] = cv2.equalizeHist(img_copy[:,:,i])
        processed_images.append((f"hist_eq_{channel_name}", img_copy))
    
    # LAB histogram equalization
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
    hist_eq_lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    processed_images.append(("hist_eq_lab", hist_eq_lab))
    
    # CLAHE variations
    clahe_configs = [(2.0, (8,8)), (4.0, (8,8)), (3.0, (16,16))]
    for clip_limit, tile_size in clahe_configs:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        processed_images.append((f"clahe_{clip_limit}_{tile_size[0]}x{tile_size[1]}", clahe_img))
    
    # Gamma corrections
    gamma_values = [0.5, 0.7, 1.3, 1.8, 2.2]
    for gamma in gamma_values:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(image, table)
        processed_images.append((f"gamma_{gamma}", gamma_corrected))
    
    # Edge enhancement
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    processed_images.append(("sharpened", sharpened))
    
    # Bilateral filter for noise reduction while preserving edges
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    processed_images.append(("bilateral", bilateral))
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph_open = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    processed_images.append(("morph_open", morph_open))
    
    return processed_images

def is_valid_plate_detection_relaxed(box, image_shape):
    """
    More relaxed validation for difficult cases
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    img_height, img_width = image_shape[:2]
    img_area = img_height * img_width
    
    # Very relaxed area ratio (0.0005 to 0.5)
    area_ratio = area / img_area
    if area_ratio < 0.0005 or area_ratio > 0.5:
        return False
    
    # Very relaxed aspect ratio for motorcycle plates
    aspect_ratio = width / height
    if aspect_ratio < 0.8 or aspect_ratio > 8.0:  # Allow more square shapes
        return False
    
    # Minimum size check
    if width < 20 or height < 10:
        return False
    
    return True

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
    
    # Aggressive preprocessing
    processed_images = preprocess_image_aggressive(image)
    
    all_detections = []
    debug_info = []
    
    # Try detection on all preprocessed versions with very low thresholds
    for method_name, proc_img in processed_images:
        try:
            # Very aggressive confidence thresholds
            confidence_thresholds = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
            
            for conf_thresh in confidence_thresholds:
                # Multiple inference parameters
                inference_params = [
                    {'conf': conf_thresh, 'iou': 0.45},
                    {'conf': conf_thresh, 'iou': 0.3},
                    {'conf': conf_thresh, 'iou': 0.7},
                ]
                
                for params in inference_params:
                    try:
                        results = yolo_model(proc_img, **params)[0]
                        
                        if results.boxes is not None and len(results.boxes) > 0:
                            for result in results.boxes:
                                x1, y1, x2, y2 = result.xyxy[0].cpu().numpy().astype(int)
                                confidence = float(result.conf[0].cpu().numpy())
                                
                                # Use relaxed validation
                                if is_valid_plate_detection_relaxed([x1, y1, x2, y2], image.shape):
                                    detection_info = {
                                        'box': [x1, y1, x2, y2],
                                        'confidence': confidence,
                                        'method': method_name,
                                        'conf_threshold': conf_thresh,
                                        'iou_threshold': params['iou']
                                    }
                                    all_detections.append(detection_info)
                                    debug_info.append(f"Detection found: {method_name}, conf={confidence:.3f}")
                    except Exception as e:
                        debug_info.append(f"Error with {method_name} conf={conf_thresh}: {str(e)}")
                        continue
        except Exception as e:
            debug_info.append(f"Error processing {method_name}: {str(e)}")
            continue
    
    # Save debug info
    debug_path = os.path.join(output_dir, "debug_log.txt")
    with open(debug_path, 'w') as f:
        f.write(f"Total detections found: {len(all_detections)}\n")
        f.write(f"Image shape: {image.shape}\n")
        f.write(f"Vehicle type: {vehicle_type}\n\n")
        for info in debug_info:
            f.write(info + "\n")
    
    if not all_detections:
        return response_api(400, 'Error', 'No plate detected', 
                          f'No plate was detected in the image. Check debug log at: {debug_path}')
    
    # Remove duplicates (same area detections)
    unique_detections = []
    for detection in all_detections:
        x1, y1, x2, y2 = detection['box']
        area = (x2-x1) * (y2-y1)
        
        is_duplicate = False
        for unique_det in unique_detections:
            ux1, uy1, ux2, uy2 = unique_det['box']
            unique_area = (ux2-ux1) * (uy2-uy1)
            
            # Check if areas are similar and boxes overlap significantly
            area_diff = abs(area - unique_area) / max(area, unique_area)
            if area_diff < 0.2:  # Similar area
                overlap_x = max(0, min(x2, ux2) - max(x1, ux1))
                overlap_y = max(0, min(y2, uy2) - max(y1, uy1))
                overlap_area = overlap_x * overlap_y
                overlap_ratio = overlap_area / min(area, unique_area)
                if overlap_ratio > 0.7:  # High overlap
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_detections.append(detection)
    
    # Sort by confidence and take the best detection
    unique_detections.sort(key=lambda x: x['confidence'], reverse=True)
    best_detection = unique_detections[0]
    
    x1, y1, x2, y2 = best_detection['box']
    h, w = image.shape[:2]

    # Dynamic margin
    detection_width = x2 - x1
    detection_height = y2 - y1
    margin_x = max(15, int(detection_width * 0.15))
    margin_y = max(15, int(detection_height * 0.15))
    
    crop_x1 = max(x1 - margin_x, 0)
    crop_y1 = max(y1 - margin_y, 0)
    crop_x2 = min(x2 + margin_x, w - 1)
    crop_y2 = min(y2 + margin_y, h - 1)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    detect_plate_color = 'UNKNOWN'

    if save_visualization:
        # Create multiple visualizations
        vis_crop = crop.copy()
        vis_full = image.copy()
        
        # Draw bounding box on crop
        box_top_left = (x1 - crop_x1, y1 - crop_y1)
        box_bottom_right = (x2 - crop_x1, y2 - crop_y1)
        cv2.rectangle(vis_crop, box_top_left, box_bottom_right, (0, 255, 0), 3)
        
        # Draw bounding box on full image
        cv2.rectangle(vis_full, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Generate filenames
        base_name = os.path.splitext(filename)[0]
        crop_filename = f"{base_name}_detected_crop.jpg"
        full_filename = f"{base_name}_detected_full.jpg"
        original_filename = f"{base_name}_original.jpg"
        
        # Save files
        crop_path = os.path.join(output_dir, crop_filename)
        full_path = os.path.join(output_dir, full_filename)
        original_path = os.path.join(output_dir, original_filename)
        
        cv2.imwrite(crop_path, vis_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(full_path, vis_full, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(original_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        

    return response_api(
        200, 'Success', 'Plate detected successfully',
        {
            'plate_type': vehicle_type,
            'plate_color': detect_plate_color,
            'plate_number': 'UNKNOWN',
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
    with open('/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/Testing Apps/Storage/test-gambar/gambar14.jpg', "rb") as f:
        img_bytes = f.read()
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        b64_input = f"data:image/jpeg;base64,{b64_str}"
    
    with delete_debug_yolo():
        result_b64 = detect_plate(b64_input, "motorcycle")

    result_b64.pop('debug', None)
    print(json.dumps(result_b64, indent=4))