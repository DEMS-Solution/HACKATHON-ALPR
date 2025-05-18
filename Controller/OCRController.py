import cv2
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import Counter

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
    """Convert to grayscale and enhance contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def detect_plate_color(plate_img):
    """Deteksi warna latar belakang dominan pada plat dengan filtering noise teks berdasarkan brightness dan saturasi."""

    h_img, w_img = plate_img.shape[:2]

    # Fokus pada area tengah plat (hindari tepi)
    x_start = int(w_img * 0.2)
    y_start = int(h_img * 0.2)
    x_end = int(w_img * 0.8)
    y_end = int(h_img * 0.8)
    cropped = plate_img[y_start:y_end, x_start:x_end]

    # Konversi ke HSV
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Buat masking: hanya ambil pixel dengan brightness cukup tinggi dan saturasi sedang
    mask = (v > 80) & (s > 40)  # Filter untuk latar belakang, abaikan teks hitam
    h_vals = h[mask]
    s_vals = s[mask]
    v_vals = v[mask]

    # Jika mask kosong (misal semua teks gelap), fallback analisa semua
    if len(h_vals) < 50:
        h_vals = h.flatten()
        s_vals = s.flatten()
        v_vals = v.flatten()

    # Gunakan median atau mode sebagai representasi
    h_median = np.median(h_vals)
    s_median = np.median(s_vals)
    v_median = np.median(v_vals)

    # Klasifikasi berdasarkan kombinasi HSV
    if v_median < 70 and s_median < 60:
        return 'BLACK'
    elif v_median > 180 and s_median < 40:
        return 'WHITE'
    elif 15 < h_median < 35 and s_median > 100:
        return 'YELLOW'
    elif (0 <= h_median < 10 or h_median > 160) and s_median > 100:
        return 'RED'
    elif 35 < h_median < 85 and s_median > 80:
        return 'GREEN'
    elif 90 < h_median < 130 and s_median > 80:
        return 'BLUE'
    else:
        return 'UNKNOWN'

def detect_plate(image_path):
    image = cv2.imread(image_path)
    results = yolo_model(image)[0]

    best_plate = ''
    plate_type = 'UNKNOWN'
    plate_color = 'UNKNOWN'

    for result in results.boxes:
        cls_id = int(result.cls[0])
        box = result.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]

        color = detect_plate_color(crop)  # Tambahan deteksi warna

        crop_preprocessed = preprocess_for_ocr(crop)
        ocr_result = ocr.ocr(crop_preprocessed, cls=True)

        if ocr_result and ocr_result[0]:
            text, conf = ocr_result[0][0][1][0], ocr_result[0][0][1][1]
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            if not text or any(re.match(p, text) for p in EXCLUDE_PATTERNS):
                continue

            if len(text) >= 5 and conf > 0.6:
                best_plate = text
                plate_color = color

                for type_name, pattern in SPECIAL_PLATES.items():
                    if re.match(pattern, best_plate):
                        plate_type = type_name
                        break
                if plate_type == 'UNKNOWN':
                    plate_type = 'CIVIL'
                break

    return best_plate if best_plate else 'UNKNOWN', plate_type, plate_color
