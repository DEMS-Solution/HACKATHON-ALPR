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
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(os.getenv('UPLOAD_FOLDER'), filename)
    file.save(filepath)

    plate_number, plate_type, plate_color = detect_plate(filepath)

    last_detection = Detection.query.filter_by(plate_number=plate_number, type=plate_type, color=plate_color) \
                                    .order_by(Detection.timestamp.desc()).first()

    if plate_number == "UNKNOWN":
        return response_api(400, 'Error', 'No plate detected', 'Plate number not detected')

    if last_detection:
        return response_api(200, 'Success', 'Deteksi Plat Nomor, Berhasil !', {
            'plate_number': plate_number,
            'type': plate_type,
            'color': plate_color,
        })

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

    return response_api(200, 'Success', 'Deteksi Plat Nomor, Berhasil !', {
        'plate_number': plate_number,
        'type': plate_type,
        'color': plate_color,
    })


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
