import cv2
import numpy as np
import re
import os
import json

from ultralytics import YOLO
from paddleocr import PaddleOCR
from Controller.PreProcessController import process_image
from Controller.YOLOController import detect_plate
from Controller.Helpers.Helper import response_api

# Load model YOLO dan PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch', drop_score=0.3)

SPECIAL_PLATES = {
    'MILITARY': r'^\d{4}-\d{2}$',
    'POLICE': r'^\d{4}-\d{2}$',
    'DUMMY': r'^XX\d{3,4}ZZ$',
    'CIVIL': r'^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$'
}


EXCLUDE_PATTERNS = [
    r'^\d{2}[:.]\d{2}$',
    r'^\d{2}/\d{2}$',
]

def preprocess_for_ocr(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tingkatkan kontras secara natural
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Blur ringan untuk hilangkan noise, tanpa hilangkan karakter
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0) 
    # preprocessed_path = image_path.replace('.jpg', '_preprocessed.jpg')
    # cv2.imwrite(preprocessed_path, blurred)
    # Jangan thresholding keras di sini!
    return blurred


def resize_image_if_needed(image):
    h, w = image.shape[:2]
    if w < 300:
        scale = 300 / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    # preprocessed_path = image.replace('.jpg', '_resize.jpg')
    # cv2.imwrite(preprocessed_path, image)
    return image

def detect_plate_ocr(image_path, plate_color, plate_type):
    image = cv2.imread(image_path)
    image = resize_image_if_needed(image)
    result = ocr.ocr(image, cls=True)

    if not result:
        return {
            'success': False,
            'message': 'Tidak ada teks yang terdeteksi oleh OCR.',
            'plate_text': [],
            'color': plate_color,
            'type': 'UNKNOWN'
        }

    plate_texts = []
    detected_type = 'REGULAR'  # default

    for line in result:
        if line is None:
            continue
        for box in line:
            text, confidence = box[1][0], box[1][1]
            text = text.strip().replace(" ", "").upper()
            if confidence > 0.5 and len(text) >= 5:
                if not any(re.match(pattern, text) for pattern in EXCLUDE_PATTERNS):
                    plate_texts.append(text)

    best_plate = None
    plate_color = plate_color.strip().lower()

    if plate_texts:
        best_plate = max(plate_texts, key=len)
        if re.match(SPECIAL_PLATES['MILITARY'], best_plate) and plate_color == 'merah kuning':
            detected_type = 'MILITARY'
        elif re.match(SPECIAL_PLATES['POLICE'], best_plate) and plate_color == 'hitam kuning':
            detected_type = 'POLICE'
        elif re.match(SPECIAL_PLATES['CIVIL'], best_plate) and plate_color == 'hitam putih':
            detected_type = 'CIVIL'
        elif re.match(SPECIAL_PLATES['DUMMY'], best_plate) and plate_color == 'hitam putih':
            detected_type = 'DUMMY'
        else:
            detected_type = 'UNKNOWN'

        return {
            'success': True,
            'plate_number': best_plate,
            'plate_text': plate_texts,
            'color': plate_color,
            'type': detected_type
        }
    else:
        return {
            'success': False,
            'message': 'OCR berhasil, tapi tidak menemukan teks plat yang cocok.',
            'plate_text': [],
            'color': plate_color,
            'type': 'UNKNOWN'
        }
    