import uuid, os
import hmac

from flask import Blueprint, request
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from datetime import datetime

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

ALLOWED_FIELDS = {'username', 'password'}

@api.route('/api/v1.0/login', methods=['POST'])
def login():
    if not request.is_json:
        return response_api(400, 'Error', 'Invalid input', 'Request must be in JSON format')

    data = request.get_json()

    extra_fields = set(data.keys()) - ALLOWED_FIELDS
    if extra_fields:
        return response_api(400, 'Error', 'Unexpected fields', f'Invalid fields in request: {", ".join(extra_fields)}')

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return response_api(400, 'Error', 'Missing credentials', 'Username and password are required')

    stored_password = USERS.get(username)
    if stored_password and hmac.compare_digest(stored_password, password):
        access_token = create_access_token(identity=username)
        return response_api(200, 'Success', 'Login successful', {'access_token': access_token})

    return response_api(401, 'Error', 'Invalid credentials', 'Invalid username or password')


@api.route('/api/v1.0/detection', methods=['POST'])
def upload_image():
    if not request.is_json:
        return response_api(
            400,
            'Error',
            'Invalid input format.',
            'Request content-type must be application/json.'
        )

    data = request.get_json()

    if not data:
        return response_api(
            400,
            'Error',
            'Invalid or missing JSON body.',
            'Expected JSON with fields: image_path and vehicle_type.'
        )

    allowed_fields = {'image_path', 'vehicle_type'}
    extra_fields = set(data.keys()) - allowed_fields
    if extra_fields:
        return response_api(
            400,
            'Error',
            'Unexpected fields in request body.',
            f'Fields not allowed: {", ".join(extra_fields)}.'
        )

    # Check for missing or empty fields
    missing_fields = [field for field in allowed_fields if not data.get(field)]
    if missing_fields:
        return response_api(
            400,
            'Error',
            f'Missing required field(s): {", ".join(missing_fields)}.',
            f'Please provide valid values for: {", ".join(missing_fields)}.'
        )

    vehicle_type = data['vehicle_type']
    image_path = data['image_path']
    bypass_detect = data.get('bypass_detect', False) 
    
    if vehicle_type not in ['car', 'motorcycle']:
        return response_api(400, 'Error', 'Invalid vehicle type', 'Tipe kendaraan tidak valid.')

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
                return response_api(400, 'Error', result['message'], 'Plat nomor tidak terdeteksi.')
            
            plate_number = result.get("plate_number", "UNKNOWN")
            if plate_number == "UNKNOWN":
                return response_api(400, 'Error', 'Plat nomor tidak terdeteksi.', 'Gagal melakukan OCR')
            
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
            return response_api(400, 'Error', result['message'], 'Plat nomor tidak terdeteksi.')
        
        plate_number = result.get("plate_number", "UNKNOWN")
        if plate_number == "UNKNOWN":
            return response_api(400, 'Error', 'Plat nomor tidak terdeteksi.', 'Gagal melakukan OCR')
            
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
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        pagination = Detection.query.order_by(Detection.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
        detections = pagination.items

        result = {
            'data': [d.to_dict() for d in detections],
            'pagination': {
                'page': pagination.page,
                'per_page': pagination.per_page,
                'total_items': pagination.total,
                'total_pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev,
            }
        }

        if not result['data']:
            return response_api(400, 'Error', 'No history found', 'No Data')

        return response_api(200, 'Success', 'History retrieved successfully', result)

    except Exception as e:
        return response_api(500, 'Error', 'Internal server error', str(e))


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
