from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from datetime import datetime
import os, uuid
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from Config.db import db
from flask_sqlalchemy import SQLAlchemy

from Models.PlatNomor import Detection
from Controller.OCRController import detect_plate
from Controller.Helpers.Helper import response_api
from Config.app import create_app
from Routes.api import api

app = create_app()
app.register_blueprint(api)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)