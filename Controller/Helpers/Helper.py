from flask import jsonify

def response_api(code=200, status='success', message='OK', data=None):
    return jsonify({
        'responseCode': code,
        'responseStatus': status,
        'responseMessage': message,
        'responseDetails': data
    }), code
