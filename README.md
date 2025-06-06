# 🚗 OpenTheGate Hack 2025

Selamat datang di proyek **OpenTheGate**, sebuah solusi ALPR (Automatic License Plate Recognition) untuk **OpenTheGate Hack 2025**, dalam rangka memperingati 10 tahun perjalanan **BSS Parking**.

Hackathon ini bertemakan:

> _“Inovasi Teknologi Parkir Berbasis ALPR”_  
> Buka gerbang inovasi. OpenTheGate. Evolve the future of parking.

---

## 🎯 Tujuan Hackathon

- ✅ Membangun sistem **ALPR berbasis AI/CV (Computer Vision)**.
- ✅ Mengelola hasil ALPR secara **terstruktur** (entry log, histori, validasi).
- ✅ Menyimulasikan **integrasi sistem parkir**:
  - Gerbang otomatis terbuka.
  - Validasi kendaraan.
  - Monitoring kendaraan secara real-time.
- ✅ Mendeteksi **format plat khusus**:
  - Plat militer.
  - Plat polisi.
  - Plat uji coba / dummy.

---

## 🛠️ Dikembangkan oleh: **DEMS Solution**

> 🔧 Tim **DEMS Solution** berkomitmen menghadirkan inovasi nyata dalam dunia perparkiran melalui pendekatan AI, rekayasa perangkat lunak, dan integrasi sistem modern.

Kami percaya bahwa masa depan parkir adalah **otomatis, aman, dan cerdas** — dan solusi ini adalah langkah nyata menuju visi tersebut.

---

## 🚀 Fitur Utama

- 🔍 Deteksi plat nomor kendaraan dengan **YOLOv8** dan **PaddleOCR**.
- 🔐 Autentikasi berbasis **JWT Token**.
- 📸 Upload gambar plat via API.
- 🧠 Deteksi dan klasifikasi **jenis & warna plat**.
- 📜 History deteksi lengkap dengan validasi.
- 🟢 Simulasi pembukaan gerbang parkir jika plat valid.

---

## 🏗️ Arsitektur Teknologi

| Komponen      | Teknologi                 |
| ------------- | ------------------------- |
| Backend       | Python, Flask             |
| OCR & Deteksi | YOLOv8, PaddleOCR, OpenCV |
| Database      | MySQL                     |
| ORM           | SQLAlchemy                |
| Auth          | Flask-JWT-Extended        |
| Konfigurasi   | dotenv                    |
| Deployment    | Gunicorn / Flask CLI      |

---

## 📁 Struktur Proyek

```bash
.
├── Config/
│ ├── Yolo/
│ │ ├── app.py
│ │ ├── db.py
│ └── requirements.txt
│
├── Controller/
│ ├── Helpers/
│ └── OCRController.py
│ └── PreProcessController.py
│ └── YOLOController.py
│
├── Models/
│ └── PlatNomor.py
│
├── Routes/
│ └── api.py
│
├── Storage/
│ └── Uploads/
│
├── .env
├── .env.example
├── .gitattributes
├── .gitignore
├── index.py
└── README.md
```
