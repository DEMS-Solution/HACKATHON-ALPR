from flask import Blueprint, request
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid, os

from Config.db import db
from Models.PlatNomor import Detection
from Controller.YOLOController import detect_plate
from Controller.PreProcessController import process_image
from Controller.OCRController import detect_plate_ocr
from Controller.Helpers.Helper import response_api

api = Blueprint('api', __name__)

USERS = {
    "admin": "password123"
}

@api.route('/api/v1.0/login', methods=['POST'])
def login():
    username = request.json.get("username")
    password = request.json.get("password")

    if USERS.get(username) == password:
        access_token = create_access_token(identity=username)
        return response_api(200, 'Success', 'Login successful', {'access_token': access_token})
    return response_api(401, 'Error', 'Invalid credentials', 'Invalid username or password')


@api.route('/api/v1.0/detection', methods=['POST'])
def upload_image():
    data = request.get_json()
    if not data:
        return response_api(400, 'Error', 'Invalid or missing JSON body.', 'Expected JSON with "vehicle_type" and "image_path".')
    
    # Validasi keberadaan field wajib
    missing_fields = []
    if 'vehicle_type' not in data or not data.get('vehicle_type'):
        missing_fields.append('vehicle_type')
    if 'image_path' not in data or not data.get('image_path'):
        missing_fields.append('image_path')
    
    if missing_fields:
        return response_api(
            400,
            'Error',
            f'Missing required field(s): {", ".join(missing_fields)}.',
            f'Harap sertakan field: {", ".join(missing_fields)} dalam body JSON.'
        )
    
    vehicle_type = data['vehicle_type']
    if vehicle_type not in ['car', 'motorcycle']:
        return response_api(400, 'Error', 'Invalid vehicle type.', 'Vehicle type harus berupa "car" atau "motorcycle".')
    image_path = data['image_path']
    bypass_detect = data.get('bypass_detect', False) 
    
    if bypass_detect:
        try:
            process_result = process_image(image_path)
            
            if hasattr(process_result, 'status_code'):
                return process_result
            
            if not isinstance(process_result, tuple) or len(process_result) != 3:
                return response_api(500, 'Error', 'Invalid response from process_image', 
                                  f'Expected tuple with 3 elements, got {type(process_result)}')
            
            detectPath, warna_plat, tipe_plat = process_result
            
            result = detect_plate_ocr(detectPath, warna_plat, tipe_plat)
            if not result['success']:
                return response_api(400, 'Error', result['message'], None)
            
            plate_number = result.get("plate_number", "UNKNOWN")
            if plate_number == "UNKNOWN":
                return response_api(400, 'Error', 'Plat nomor tidak terdeteksi.', None)
            
            # Cek data terakhir di DB
            last_detection = Detection.query.filter_by(
                plate_number=plate_number,
                type=tipe_plat,
                color=warna_plat
            ).order_by(Detection.timestamp.desc()).first()
            
            if last_detection:
                return response_api(200, 'Success', 'Deteksi plat nomor berhasil.', {
                    'plate_number': plate_number,
                    'type': tipe_plat,
                    'color': warna_plat,
                    'detected_path': full_path,
                    'original_path': original_path,
                    'crop_path': platDetectPath,
                })
            
            # Simpan deteksi baru
            detection = Detection(
                id=uuid.uuid4(),
                plate_number=plate_number,
                image_path=full_path,
                type=tipe_plat,
                color=warna_plat,
                timestamp=datetime.now(),
                is_validated=False
            )
            db.session.add(detection)
            db.session.commit()
            
            return response_api(200, 'Success', 'Plat nomor berhasil disimpan.', {
                'plate_number': plate_number,
                'type': tipe_plat,
                'color': warna_plat,
                'detected_path': full_path,
                'original_path': original_path,
                'crop_path': platDetectPath,
            })
            
        except Exception as e:
            return response_api(500, 'Error', 'Error in bypass detect flow', str(e))
    
    try:
        detect_result = detect_plate(image_path, vehicle_type)
        
        if hasattr(detect_result, 'status_code'):
            platDetectPath = image_path
            
        elif isinstance(detect_result, tuple) and len(detect_result) == 4:
            platDetectPath, full_path, original_path, vehicle_type = detect_result
            
            if not isinstance(platDetectPath, str):
                return response_api(500, 'Error', 'Invalid plate detection path', 
                                  f'Expected string path, got {type(platDetectPath)}')
        else:
            return response_api(500, 'Error', 'Invalid response from detect_plate', 
                              f'Gambar tidak ditemukan: {image_path}')
    except Exception as detect_error:
        print(f"DEBUG: Exception in detect_plate: {str(detect_error)}")
        return response_api(500, 'Error', 'Error calling detect_plate', str(detect_error))
    
    if not os.path.exists(platDetectPath):
        return response_api(400, 'Error', 'File tidak ditemukan', 
                          f'File hasil detect_plate tidak ditemukan: {platDetectPath}')
    
    try:
        process_result = process_image(platDetectPath)
        
        if hasattr(process_result, 'status_code'):
            return process_result
        
        if not isinstance(process_result, tuple) or len(process_result) != 3:
            return response_api(500, 'Error', 'Invalid response from process_image', 
                              f'Expected tuple with 3 elements, got {type(process_result)}')
        
        detectPath, warna_plat, tipe_plat = process_result
        
        if not isinstance(detectPath, str):
            return response_api(500, 'Error', 'Invalid processed image path', 
                              f'Expected string path, got {type(detectPath)}')
    except Exception as process_error:
        return response_api(500, 'Error', 'Error calling process_image', str(process_error))
    
    try:
        result = detect_plate_ocr(detectPath, warna_plat, tipe_plat)
        if not result['success']:
            return response_api(400, 'Error', result['message'], None)
        
        plate_number = result.get("plate_number", "UNKNOWN")
        if plate_number == "UNKNOWN":
            return response_api(400, 'Error', 'Plat nomor tidak terdeteksi.', None)
            
    except Exception as ocr_error:
        return response_api(500, 'Error', 'Error in OCR detection', str(ocr_error))
    
    try:
        last_detection = Detection.query.filter_by(
            plate_number=plate_number,
            type=tipe_plat,
            color=warna_plat
        ).order_by(Detection.timestamp.desc()).first()
        
        if last_detection:
            return response_api(200, 'Success', 'Deteksi plat nomor berhasil.', {
                'plate_number': plate_number,
                'type': tipe_plat,
                'color': warna_plat,
                'detected_path': full_path,
                'original_path': original_path,
                'crop_path': platDetectPath,
            })
            
    except Exception as db_query_error:
        return response_api(500, 'Error', 'Database query error', str(db_query_error))
    
    try:
        detection = Detection(
            id=uuid.uuid4(),
            plate_number=plate_number,
            image_path=full_path,
            type=tipe_plat,
            color=warna_plat,
            timestamp=datetime.now(),
            is_validated=False
        )
        
        db.session.add(detection)
        db.session.commit()
        
    except Exception as db_error:
        return response_api(500, 'Error', 'Database error', str(db_error))
    
    return response_api(200, 'Success', 'Plat nomor berhasil disimpan.', {
        'plate_number': plate_number,
        'type': tipe_plat,
        'color': warna_plat,
        'detected_path': full_path,
        'original_path': original_path,
        'crop_path': platDetectPath,
    }) 
    
@api.route('/api/v1.0/history', methods=['GET'])
def get_history():
    detections = Detection.query.order_by(Detection.timestamp.desc()).all()
    return response_api(200, 'Success', 'History retrieved successfully',
                        [d.to_dict() for d in detections])


@api.route('/api/v1.0/validate', methods=['POST'])
def validate_plate():
    plate = request.json.get('plate_number')
    detection = Detection.query.filter_by(plate_number=plate).order_by(Detection.timestamp.desc()).first()
    if detection:
        detection.is_validated = True
        db.session.commit()
        return response_api(200, 'Success', 'Plate validated successfully', detection.to_dict())
    return response_api(404, 'Error', 'Plate not found')


@api.route('/api/v1.0/gate-status/<plate>', methods=['GET'])
def gate_status(plate):
    detection = Detection.query.filter_by(plate_number=plate).order_by(Detection.timestamp.desc()).first()
    if detection and detection.is_validated:
        return response_api(200, 'Success', 'Gate opened', {'gate': 'open'})
    return response_api(403, 'Error', 'Gate closed', {'gate': 'closed'})
