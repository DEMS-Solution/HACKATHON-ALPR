from flask import Blueprint, request
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid, os

from Config.db import db
from Models.PlatNomor import Detection
from Controller.OCRController import detect_plate
from Controller.Helpers.Helper import response_api

api = Blueprint('api', __name__)

USERS = {
    "admin": "password123"
}

@api.route('/api/login', methods=['POST'])
def login():
    username = request.json.get("username")
    password = request.json.get("password")

    if USERS.get(username) == password:
        access_token = create_access_token(identity=username)
        return response_api(200, 'Success', 'Login successful', {'access_token': access_token})
    return response_api(401, 'Error', 'Invalid credentials', 'Invalid username or password')


@api.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return response_api(400, 'Error', 'No image provided', 'Form field "image" is required.')

    file = request.files['image']
    if file.filename == '':
        return response_api(400, 'Error', 'No image selected', 'File name is empty.')

    filename = secure_filename(file.filename)
    upload_folder = os.getenv('UPLOAD_FOLDER', 'uploads')  # fallback jika env tidak diset
    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    try:
        # Call ALPR Detection
        plate_number, plate_type, plate_color, plate_box = detect_plate(filepath, save_visualization=True)

        # Path untuk hasil visualisasi YOLO
        viz_path = os.path.splitext(filepath)[0] + "_detected.jpg"

        if plate_number == "UNKNOWN":
            return response_api(400, 'Error', 'No plate detected', 'Plat nomor tidak terdeteksi.')

        # Cek apakah plat nomor sudah pernah terdeteksi sebelumnya
        last_detection = Detection.query.filter_by(
            plate_number=plate_number,
            type=plate_type,
            color=plate_color
        ).order_by(Detection.timestamp.desc()).first()

        if last_detection:
            return response_api(200, 'Success', 'Deteksi plat nomor berhasil disimpan.', {
                'plate_number': plate_number,
                'type': plate_type,
                'color': plate_color,
                'visualization_path': viz_path
            })

        # Simpan deteksi baru ke DB
        detection = Detection(
            id=uuid.uuid4(),
            plate_number=plate_number,
            image_path=filepath,
            type=plate_type,
            color=plate_color,
            timestamp=datetime.now(),
            is_validated=False
        )
        db.session.add(detection)
        db.session.commit()

        return response_api(200, 'Success', 'Deteksi plat nomor berhasil disimpan.', {
            'plate_number': plate_number,
            'type': plate_type,
            'color': plate_color,
            'visualization_path': viz_path
        })

    except Exception as e:
        return response_api(500, 'Error', 'Internal Server Error', str(e))
    
    
@api.route('/api/history', methods=['GET'])
def get_history():
    detections = Detection.query.order_by(Detection.timestamp.desc()).all()
    return response_api(200, 'Success', 'History retrieved successfully',
                        [d.to_dict() for d in detections])


@api.route('/api/validate', methods=['POST'])
def validate_plate():
    plate = request.json.get('plate_number')
    detection = Detection.query.filter_by(plate_number=plate).order_by(Detection.timestamp.desc()).first()
    if detection:
        detection.is_validated = True
        db.session.commit()
        return response_api(200, 'Success', 'Plate validated successfully', detection.to_dict())
    return response_api(404, 'Error', 'Plate not found')


@api.route('/api/gate-status/<plate>', methods=['GET'])
def gate_status(plate):
    detection = Detection.query.filter_by(plate_number=plate).order_by(Detection.timestamp.desc()).first()
    if detection and detection.is_validated:
        return response_api(200, 'Success', 'Gate opened', {'gate': 'open'})
    return response_api(403, 'Error', 'Gate closed', {'gate': 'closed'})
