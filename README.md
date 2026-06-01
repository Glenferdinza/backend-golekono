# Golekono ML Backend

Layanan backend berbasis Django REST Framework untuk aplikasi Golekono. Backend ini berfungsi mengintegrasikan model Machine Learning untuk analisis sentimen suasana liburan dan sistem rekomendasi itinerary perjalanan wisata Yogyakarta berbasis Content-Based Filtering.

## Persyaratan Sistem

- Python 3.10 atau versi di atasnya (Rekomendasi: Python 3.13)
- Docker dan Docker Compose (untuk menjalankan Prometheus dan Grafana)

## Struktur Folder Utama

- `api/`: Berisi views, urls, dan apps konfigurasi untuk endpoint Machine Learning.
- `config/`: Konfigurasi proyek Django utama (settings, routing, dll).
- `ml_model/`: Folder berisi dataset `wisata_clean.csv` dan model-model Machine Learning (`.joblib`, `.npy`, `.pkl`).
- `monitoring/`: Konfigurasi Prometheus untuk scraping metrics.
- `docker-compose.yml`: Berisi konfigurasi orkestrasi container untuk Prometheus dan Grafana.

## Cara Menjalankan Aplikasi Secara Lokal

1. Masuk ke direktori backend-golekono:
   ```bash
   cd backend-golekono
   ```

2. Aktifkan virtual environment:
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

3. Persiapkan database MySQL:
   - Pastikan MySQL server Anda aktif (misalnya melalui XAMPP, Laragon, atau instalasi native) pada port `3306`.
   - Buat database baru bernama `golekono_db` (misalnya dengan menjalankan `CREATE DATABASE golekono_db;` di phpMyAdmin atau MySQL CLI).

4. Jalankan migrasi database MySQL Django:
   ```bash
   py manage.py migrate
   ```

5. Jalankan server pembangunan Django:
   ```bash
   py manage.py runserver 0.0.0.0:8000
   ```

Server backend akan berjalan di http://localhost:8000. Endpoint rekomendasi utama berada di http://localhost:8000/api/itinerary/.

## Cara Menjalankan Monitoring MLOps (Prometheus & Grafana)

Aplikasi ini menggunakan Prometheus untuk melacak data metrik kinerja model ML dan server, serta Grafana untuk visualisasi dashboard monitoring.

1. Jalankan docker compose di folder backend-golekono:
   ```bash
   docker compose up -d
   ```

2. Akses layanan:
   - **Prometheus**: http://localhost:9092 (Menampilkan metrik mentah dan status scrape backend)
   - **Grafana**: http://localhost:3001 (Untuk visualisasi metrik dengan dashboard)
     - Username default: `admin`
     - Password default: `admin`

## Metrik Kustom MLOps

Aplikasi Django ini mengekspos beberapa metrik kustom di endpoint `/metrics` yang dapat dipantau langsung:
- `ml_itinerary_generation_latency_seconds`: Melacak latensi/waktu komputasi yang dihabiskan untuk menghasilkan rekomendasi itinerary dan inferensi model ML.
- `ml_sentiment_classifications_total`: Menghitung frekuensi klasifikasi sentimen berdasarkan label hasil inferensi ("Positif", "Negatif", "Netral", "None").
- `ml_itineraries_generated_total`: Melacak jumlah akumulatif itinerary wisata yang sukses digenerasi oleh sistem.
