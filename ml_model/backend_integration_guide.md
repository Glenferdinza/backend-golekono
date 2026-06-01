# 🚀 Panduan Integrasi ML ke Backend Django
**Dokumen Serah Terima (Handoff) dari Tim ML ke Tim Backend**

Tim Backend, berikut adalah file model Machine Learning dan panduan cara mengimplementasikannya ke dalam *endpoint* API Django Anda.

## 📦 1. File yang Harus Disalin ke Project Django
Buat folder baru bernama `ml_models/` di dalam *root directory* project Django Anda, lalu salin folder & file berikut dari tim ML:

1. **Folder `models/`** (Berisi 9 file model dari ML):
   - `tfidf_vectorizer.joblib`
   - `label_encoder.joblib`
   - `sentiment_logreg.joblib` *(Model Sentimen Utama)*
   - `sentiment_nn.keras` *(Opsional)*
   - `cosine_similarity_matrix.npy`
   - `cbf_scaler.joblib`
   - `cbf_ohe.joblib`
   - `cbf_feature_matrix.npy`
   - `wisata_indexed.pkl` *(Database Pandas untuk Algoritma)*

2. **File Dependencies**:
   - `requirements.txt` (Backend harus menginstall library ini di environment Django: `pip install -r requirements.txt`)

3. **Script Referensi Logika Algoritma ML**:
   - `demo_itinerary.py` *(Backend cukup menyalin/copy-paste fungsi `generate_itinerary` dari file ini)*

---

## 🛠️ 2. Cara Load Model di Django (apps.py)
**SANGAT PENTING:** Jangan me-load model di dalam `views.py` karena akan membuat API sangat lambat (model di-load dari harddisk setiap kali ada request HTTP). 
Load model **sekali saja** ke dalam memori RAM saat server Django pertama kali menyala melalui `apps.py`.

```python
# nama_app/apps.py
from django.apps import AppConfig
import os
import joblib
import pickle
import numpy as np
from django.conf import settings

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api' # sesuaikan nama app

    # Global dictionary untuk menyimpan instance model ML di RAM
    ml_models = {}

    def ready(self):
        # Path ke folder models yang diberikan tim ML
        model_dir = os.path.join(settings.BASE_DIR, 'ml_models', 'models')
        
        # 1. Load Model Sentimen
        self.ml_models['tfidf'] = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
        self.ml_models['le'] = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
        self.ml_models['logreg'] = joblib.load(os.path.join(model_dir, 'sentiment_logreg.joblib'))
        
        # 2. Load Model Rekomendasi (CBF)
        self.ml_models['cosine_sim'] = np.load(os.path.join(model_dir, 'cosine_similarity_matrix.npy'))
        
        # 3. Load DataFrame Wisata
        with open(os.path.join(model_dir, 'wisata_indexed.pkl'), 'rb') as f:
            wisata_data = pickle.load(f)
            
        self.ml_models['df_wisata'] = wisata_data['df']
        self.ml_models['name_to_idx'] = wisata_data['name_to_idx']
        
        print("✅ [ML] Semua Model Machine Learning Berhasil Di-load ke Memori Server!")
```

---

## 🌐 3. Contoh Kode Endpoint API (views.py)

Berikut cara memanggil model yang sudah di-load di memori untuk merespon *HTTP Request*.

### A. Endpoint Prediksi Sentimen Review
```python
# nama_app/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.apps import apps
import json

@csrf_exempt
def predict_sentiment(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            review_text = data.get('review', '')
            
            # Ambil model dari memori apps.py (TIDAK load dari harddisk lagi!)
            api_app = apps.get_app_config('api')
            tfidf = api_app.ml_models['tfidf']
            logreg = api_app.ml_models['logreg']
            le = api_app.ml_models['le']
            
            # Eksekusi Prediksi ML
            X = tfidf.transform([review_text])
            pred_idx = logreg.predict(X)[0]
            confidence = logreg.predict_proba(X)[0].max() * 100
            label = le.inverse_transform([pred_idx])[0]
            
            return JsonResponse({
                'status': 'success',
                'review': review_text,
                'sentiment': label,
                'confidence_pct': round(confidence, 2)
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
```

### B. Endpoint Generate Itinerary (Round-Robin)
Tim backend cukup menyalin fungsi `generate_itinerary` secara penuh dari file `demo_itinerary.py` buatan tim ML, lalu memanggilnya di dalam fungsi view.

```python
# nama_app/views.py
from django.http import JsonResponse
from django.apps import apps
import json

def generate_itinerary(df, minat, budget_per_tempat, durasi_hari, destinasi_per_hari=3):
    # [PASTE FULL KODE DARI demo_itinerary.py TIM ML DI SINI]
    # (Kode disembunyikan di dokumen ini demi keringkasan)
    return itinerary, total_biaya

@csrf_exempt
def get_itinerary(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            minat = data.get('minat', [])
            budget = int(data.get('budget', 25000))
            durasi = int(data.get('durasi', 3))
            
            # Ambil dataset Pandas wisata dari memory
            api_app = apps.get_app_config('api')
            df_wisata = api_app.ml_models['df_wisata']
            
            # Eksekusi algoritma Itinerary ML
            itinerary_result, estimasi_biaya = generate_itinerary(
                df=df_wisata,
                minat=minat,
                budget_per_tempat=budget,
                durasi_hari=durasi
            )
            
            return JsonResponse({
                'status': 'success',
                'estimasi_total_tiket': estimasi_biaya,
                'itinerary': itinerary_result
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
```
