from flask import jsonify, has_app_context
import json

def response_api(code=200, status='success', message='OK', data=None):
    payload = {
        'responseCode': code,
        'responseStatus': status,
        'responseMessage': message,
        'responseDetails': data
    }
    
    if has_app_context():
        return jsonify(payload), code
    else:
        return payload