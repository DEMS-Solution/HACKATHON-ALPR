import cv2
import numpy as np
import os
import json
from YOLOController import detect_plate
from Helpers.Helper import response_api

# Inisialisasi output directory
OUTPUT_DIR = 'Storage/Uploads'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_plate_color(plate_img):
    """
    Menganalisis warna plat nomor
    """
    # Konversi ke HSV untuk analisis warna yang lebih baik
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    
    # Definisikan range warna
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    
    # Hitung persentase warna
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    total_pixels = plate_img.shape[0] * plate_img.shape[1]
    white_percent = (np.sum(white_mask > 0) / total_pixels) * 100
    black_percent = (np.sum(black_mask > 0) / total_pixels) * 100
    yellow_percent = (np.sum(yellow_mask > 0) / total_pixels) * 100
    red_percent = (np.sum(red_mask > 0) / total_pixels) * 100
    green_percent = (np.sum(green_mask > 0) / total_pixels) * 100
    blue_percent = (np.sum(blue_mask > 0) / total_pixels) * 100
    
    # Buat dictionary warna dan persentasenya
    color_percentages = {
        'white': white_percent,
        'black': black_percent,
        'yellow': yellow_percent,
        'red': red_percent,
        'green': green_percent,
        'blue': blue_percent
    }
    
    # Urutkan warna berdasarkan persentase
    sorted_colors = sorted(color_percentages.items(), key=lambda x: x[1], reverse=True)
    
    # Ambil dua warna dominan
    dominant_colors = {
        'primary': {'color': sorted_colors[0][0], 'percentage': sorted_colors[0][1]},
        'secondary': {'color': sorted_colors[1][0], 'percentage': sorted_colors[1][1]}
    }
    
    # Gabungkan hasil
    result = {
        'all_colors': color_percentages,
        'dominant_colors': dominant_colors
    }
    
    return result

def classify_plate_type(color_analysis):
    """
    Mengklasifikasikan tipe plat berdasarkan analisis warna
    """
    # Ambil persentase warna dari all_colors
    colors = color_analysis['all_colors']
    red_percent = colors['red']
    yellow_percent = colors['yellow']
    black_percent = colors['black']
    white_percent = colors['white']
    
    # Kategori Militer:
    # - Merah > 3%
    # - Kuning > 0.02%
    # - Hitam dan Putih < 10%
    if (red_percent > 3.0 and 
        yellow_percent > 0.02 and 
        black_percent < 10.0 and 
        white_percent < 10.0):
        return "Military"
    
    # Kategori Polisi:
    # - Kuning > 2%
    # - Hitam > 5%
    elif yellow_percent > 2.0 and black_percent > 5.0:
        return "Police"
    
    # Kategori Civil:
    # - Hitam > 15%
    # - Putih > 15%
    elif black_percent > 15.0 and white_percent > 15.0:
        return "Civil"
    
    return "Unknown"

def preprocess_image(image, plate_type):
    """
    Melakukan preprocessing pada gambar untuk meningkatkan kualitas OCR
    """
    try:
        # Konversi ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalisasi kontras menggunakan CLAHE dengan parameter yang lebih natural
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Blur ringan untuk menghilangkan noise tanpa menghilangkan karakter
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Peningkatan ketajaman yang lebih halus
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 9.0
        sharpened = cv2.filter2D(blurred, -1, kernel)
        
        # Normalisasi intensitas
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        
        # Reduksi noise dengan bilateral filter yang lebih halus
        denoised = cv2.bilateralFilter(normalized, 5, 75, 75)
        
        if plate_type == "Police":
            # Untuk plat polisi, tingkatkan kontras kuning-hitam
            denoised = cv2.convertScaleAbs(denoised, alpha=1.2, beta=10)
            
        elif plate_type == "Military":
            # Untuk plat militer, tingkatkan kontras kuning-merah
            denoised = cv2.convertScaleAbs(denoised, alpha=1.3, beta=5)
            
        else:  # Civil
            # Untuk plat civil, pertahankan kontras natural
            denoised = cv2.convertScaleAbs(denoised, alpha=1.1, beta=0)
        
        return denoised
        
    except Exception as e:
        print(f"Error pada preprocessing: {str(e)}")
        return None

def process_image(image_path):
    """
    Memproses gambar crop dari YOLOController untuk analisis warna dan preprocessing
    """
    try:
        if not os.path.exists(image_path):
            return response_api(400, 'Error', 'File tidak ditemukan', f'File gambar tidak ditemukan di {image_path}')
            
        # Baca gambar plat yang sudah di-crop
        plate_img = cv2.imread(image_path)
        if plate_img is None:
            return response_api(400, 'Error', 'Gambar plat tidak valid', f'Tidak dapat membaca gambar plat dari {image_path}')
        
        # Analisis warna untuk menentukan tipe plat
        color_analysis = analyze_plate_color(plate_img)
        plate_type = classify_plate_type(color_analysis)
        
        plate_color = 'UNKNOWN'
        if plate_type == "Military":
            plate_color = "Merah Kuning"
        elif plate_type == "Police":
            plate_color = "Hitam Kuning"
        elif plate_type == "Civil":
            plate_color = "Hitam Putih"
        
        # Preprocessing untuk OCR dengan plate_type
        processed_img = preprocess_image(plate_img, plate_type)
        if processed_img is None:
            return response_api(400, 'Error', 'Gagal melakukan preprocessing', 'Gambar hasil preprocessing tidak valid')
            
        # Simpan hasil di folder yang sama dengan file input
        output_folder = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        grayscale_filename = f"{base_name}_grayscale.jpg"
        grayscale_path = os.path.join(output_folder, grayscale_filename)
        
        try:
            cv2.imwrite(grayscale_path, processed_img)
        except Exception as save_error:
            return response_api(400, 'Error', 'Gagal menyimpan gambar', str(save_error))
        
        return grayscale_path, plate_color, plate_type
            
    except Exception as e:
        return response_api(500, 'Error', 'Error pada process_image', str(e))

# if __name__ == "__main__":
#     try:
#         # Contoh penggunaan dengan file crop dari YOLOController
#         image_path = "C:\\laragon\\www\\HACKATHON-ALPR\\Storage\\Uploads\\img_base64\\img64_detected_crop.jpg"
#         result = process_image(image_path)
#         print(json.dumps(result, indent=4))
#     except Exception as e:
#         error_response = response_api(500, 'Error', 'Error pada test_preprocessing', str(e))
#         print(json.dumps(error_response, indent=4))
