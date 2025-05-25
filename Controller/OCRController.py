import cv2
import numpy as np
import re
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Load model YOLO dan PaddleOCR
yolo_model = YOLO('/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/Testing Apps/Config/Yolo/license_plate_detector.pt')
ocr = PaddleOCR(use_angle_cls=True, lang='en')

SPECIAL_PLATES = {
    'MILITARY': r'^T\d{1,4}[A-Z]{1,2}$',
    'POLICE': r'^B\d{1,4}PM$',
    'DUMMY': r'^XX\d{1,4}ZZ$'
}
EXCLUDE_PATTERNS = [
    r'^\d{2}[:.]\d{2}$',
    r'^\d{2}/\d{2}$',
]

def preprocess_for_ocr(img):
    """Ubah ke grayscale dan tingkatkan kontras untuk OCR"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def detect_plate_color(plate_img):
    """Deteksi warna latar dominan plat nomor."""
    h_img, w_img = plate_img.shape[:2]
    x_start, y_start = int(w_img * 0.2), int(h_img * 0.2)
    x_end, y_end = int(w_img * 0.8), int(h_img * 0.8)
    cropped = plate_img[y_start:y_end, x_start:x_end]

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (v > 80) & (s > 40)

    h_vals, s_vals, v_vals = h[mask], s[mask], v[mask]
    if len(h_vals) < 50:
        h_vals, s_vals, v_vals = h.flatten(), s.flatten(), v.flatten()

    h_median, s_median, v_median = np.median(h_vals), np.median(s_vals), np.median(v_vals)

    if v_median < 70 and s_median < 60: return 'BLACK'
    elif v_median > 180 and s_median < 40: return 'WHITE'
    elif 15 < h_median < 35 and s_median > 100: return 'YELLOW'
    elif (0 <= h_median < 10 or h_median > 160) and s_median > 100: return 'RED'
    elif 35 < h_median < 85 and s_median > 80: return 'GREEN'
    elif 90 < h_median < 130 and s_median > 80: return 'BLUE'
    else: return 'UNKNOWN'

def detect_plate(image_path, save_visualization=False):
    image = cv2.imread(image_path)
    original_image = image.copy()
    plate_only = image.copy()
    results = yolo_model(image)[0]

    best_plate, plate_type, plate_color, plate_box = '', 'UNKNOWN', 'UNKNOWN', None

    for result in results.boxes:
        cls_id = int(result.cls[0])
        box = result.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        color = detect_plate_color(crop)

        # OCR preprocessing
        crop_preprocessed = preprocess_for_ocr(crop)
        ocr_result = ocr.ocr(crop_preprocessed, cls=True)

        if ocr_result and ocr_result[0]:
            text, conf = ocr_result[0][0][1][0], ocr_result[0][0][1][1]
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            if not text or any(re.match(p, text) for p in EXCLUDE_PATTERNS):
                continue

            if len(text) >= 5 and conf > 0.6:
                best_plate, plate_color, plate_box = text, color, box
                plate_type = next((k for k, v in SPECIAL_PLATES.items() if re.match(v, text)), 'CIVIL')
                break

    # Optional visualizations
    if plate_box is not None and save_visualization:
        x1, y1, x2, y2 = plate_box
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{best_plate} ({plate_type}, {plate_color})"
        cv2.putText(original_image, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        crop_for_chars = image[y1:y2, x1:x2]
        result = ocr.ocr(crop_for_chars, cls=True)

        for line in result:
            for word in line:
                points = word[0]
                text = word[1][0]
                conf = word[1][1]
                if conf < 0.4: continue
                pts = [(int(p[0]+x1), int(p[1]+y1)) for p in points]
                for j in range(4):
                    cv2.line(plate_only, pts[j], pts[(j+1)%4], (255, 0, 0), 2)
                cv2.putText(plate_only, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        base_path, ext = os.path.splitext(image_path)
        cv2.imwrite(base_path + '_detected.jpg', original_image)
    
    return best_plate if best_plate else 'UNKNOWN', plate_type, plate_color, plate_box
