import uuid
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.types import TypeDecorator
from Config.db import db

class GUID(TypeDecorator):
    """Platform-independent GUID (UUID) type."""
    impl = CHAR

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(uuid.UUID(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return uuid.UUID(value)

class Detection(db.Model):
    __tablename__ = 'detections'

    id = db.Column(GUID(), primary_key=True, default=uuid.uuid4)
    plate_number = db.Column(db.String(20), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(20), nullable=False)
    color = db.Column(db.String)
    timestamp = db.Column(db.DateTime, nullable=False)
    is_validated = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'id': str(self.id),
            'plate_number': self.plate_number,
            'image_path': self.image_path,
            'type': self.type,
            'color': self.color,
            'timestamp': self.timestamp.isoformat(),
            'is_validated': self.is_validated
        }
