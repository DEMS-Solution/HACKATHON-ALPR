import os
import cv2
import base64
import numpy as np
import json

from PIL import Image
from io import BytesIO
from ultralytics import YOLO

from Helpers.Helper import response_api

# Load YOLO model
yolo_model = YOLO('/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/Testing Apps/Config/Yolo/best.pt')

base_dir = os.path.dirname(os.path.abspath(__file__))
upload_folder = os.getenv('UPLOAD_FOLDER', os.path.abspath(os.path.join(base_dir, '..', 'Storage', 'Uploads')))
os.makedirs(upload_folder, exist_ok=True)

def get_unique_output_path(base_name="img64", ext=".jpeg", folder=upload_folder):
    index = 1
    while True:
        filename = f"{index}_{base_name}_detected{ext}"
        output_path = os.path.join(folder, filename)
        if not os.path.exists(output_path):
            return filename, output_path
        index += 1

def detect_plate(image_input, vehicle_type):
    is_base64 = False
    save_visualization = True

    if vehicle_type not in ["car", "motorcycle"]:
        return response_api(400, 'Error', 'Invalid vehicle type', 'Only car or motorcycle are supported.')

    # Load image
    if isinstance(image_input, str):
        if os.path.isfile(image_input):
            image = cv2.imread(image_input)
            filename = os.path.basename(image_input)
        else:
            try:
                header_removed = image_input.split(',')[-1]
                img_data = base64.b64decode(header_removed)
                image_np = np.array(Image.open(BytesIO(img_data)).convert('RGB'))
                image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                is_base64 = True
                filename = "img64.jpeg"
            except Exception as e:
                return response_api(400, 'Error', 'Invalid image input', str(e))
    else:
        return response_api(400, 'Error', 'Invalid image input type', 'Input must be a file path or base64 string.')

    if image is None:
        return response_api(400, 'Error', 'Image not found or invalid', 'Could not read the image.')

    # YOLO inference
    results = yolo_model(image)[0]

    for result in results.boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy().astype(int)
        h, w = image.shape[:2]

        margin = 20
        crop_x1 = max(x1 - margin, 0)
        crop_y1 = max(y1 - margin, 0)
        crop_x2 = min(x2 + margin, w - 1)
        crop_y2 = min(y2 + margin, h - 1)

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        if save_visualization:
            box_top_left = (x1 - crop_x1, y1 - crop_y1)
            box_bottom_right = (x2 - crop_x1, y2 - crop_y1)
            cv2.rectangle(crop, box_top_left, box_bottom_right, (255, 0, 255), 2)

            if is_base64:
                base_name, ext = "img64", ".jpeg"
                filename, output_path = get_unique_output_path(base_name, ext)
            else:
                base_name, ext = os.path.splitext(filename)
                output_path = os.path.join(upload_folder, f"{base_name}_detected{ext}")

            saved = cv2.imwrite(output_path, crop)

        return response_api(
            200, 'Success', 'Plate detected successfully',
            {
                'plate_type': vehicle_type,
                'plate_color': "UNKNOWN",
                'visualization_path': output_path if save_visualization else None
            }
        )

    return response_api(400, 'Error', 'No plate detected', 'No plate was detected in the image.')

if __name__ == "__main__":
    # TEST KALAU PATH IMAGE LANGSUNG
    result = detect_plate(
        '/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/Testing Apps/Storage/test-gambar/gambar3.jpeg',"car",
    )
    print(json.dumps(result, indent=4))
    
    # TEST KALAU BASE64 INPUT
    # with open('/Users/eki/File Eki/2023 - 2024/Kerjaan/Hackathon/Testing Apps/Storage/test-gambar/gambar3.jpeg', "rb") as f:
    #     img_bytes = f.read()
    #     b64_str = base64.b64encode(img_bytes).decode("utf-8")
    #     b64_input = f"data:image/jpeg;base64,{b64_str}"

    # result_b64 = detect_plate(
    #     b64_input,
    #     vehicle_type="CAR",
    #     save_visualization=True
    # )
