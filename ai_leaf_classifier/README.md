# Tomato Leaf Healthy vs Sick — EfficientNet-B2

Project ini memakai **dataset yang sudah ada di `archive.zip`** pada folder project dan melatih **EfficientNet-B2 (PyTorch)** untuk klasifikasi biner:

- `healthy`
- `sick`

## Dataset yang terdeteksi dari archive.zip

Archive berisi struktur:

```text
tomato/
├── cnn_train.py
├── train/   # 10,000 gambar
└── val/     # 1,000 gambar
```

Ada 10 kelas asli. Masing-masing kelas memiliki 1,000 gambar pada `train` dan 100 gambar pada `val`:

- Tomato___Bacterial_spot
- Tomato___Early_blight
- Tomato___Late_blight
- Tomato___Leaf_Mold
- Tomato___Septoria_leaf_spot
- Tomato___Spider_mites Two-spotted_spider_mite
- Tomato___Target_Spot
- Tomato___Tomato_Yellow_Leaf_Curl_Virus
- Tomato___Tomato_mosaic_virus
- Tomato___healthy

Untuk target internship ini:

```text
Tomato___healthy -> healthy
9 kelas penyakit -> sick
```

Tidak perlu mengubah atau memindahkan 11,000 gambar secara manual.

## 1. Setelah folder dipindah ke VS Code

Pastikan `archive.zip` berada di root project:

```text
ai_leaf_classifier/
├── archive.zip
├── prepare_dataset.py
├── config.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── camera_inference.py
├── requirements.txt
├── raw_dataset/
├── models/
└── results/
```

## 2. Buat environment

Satu venv dipakai bersama oleh seluruh folder `Agriculture/` (termasuk demo kamera guava di root), dibuat di `Agriculture/.venv`.

Linux / macOS:

```bash
cd Agriculture
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r ai_leaf_classifier/requirements.txt
```

Windows PowerShell:

```powershell
cd Agriculture
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r ai_leaf_classifier\requirements.txt
```

Jalankan script `ai_leaf_classifier/*.py` dengan `cwd` di dalam folder `ai_leaf_classifier/` (script memakai import relatif seperti `from config import ...`), contoh:

```bash
cd ai_leaf_classifier
../.venv/bin/python prepare_dataset.py
../.venv/bin/python train.py
```

Jika GPU dipakai bersama proses lain (cek `nvidia-smi`), pilih GPU yang kosong dengan `CUDA_VISIBLE_DEVICES`, misalnya:

```bash
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python train.py
```

## 3. Extract dan cek dataset

```powershell
python prepare_dataset.py
```

Script otomatis mengekstrak `archive.zip` menjadi:

```text
raw_dataset/
└── tomato/
    ├── train/
    └── val/
```

Script juga memverifikasi jumlah kelas dan gambar.

## 4. Pembagian data yang dipakai

Dataset asli `train` (10,000 gambar) dibagi **per kelas asli** supaya setiap penyakit tetap terwakili:

- 80% -> training = 8,000 gambar
- 20% -> validation = 2,000 gambar

Dataset asli `val` dipakai sebagai **final test set** = 1,000 gambar.

Karena hanya ada 1 kelas sehat dan 9 kelas sakit, binary dataset tidak seimbang:

```text
training:   healthy 800   | sick 7,200
validation: healthy 200   | sick 1,800
test:       healthy 100   | sick 900
```

`train.py` menangani ketidakseimbangan ini menggunakan **class-weighted CrossEntropyLoss**.

## 5. Training EfficientNet-B2

```powershell
python train.py
```

Model memakai:

- `EfficientNet_B2_Weights.IMAGENET1K_V1`
- input `288 x 288`
- ImageNet normalization
- random crop, flip, rotation, dan color jitter
- AdamW optimizer
- early stopping
- weighted loss untuk imbalance

Model terbaik disimpan sebagai:

```text
models/efficientnet_b2_tomato_health.pt
```

## 6. Evaluasi

```powershell
python evaluate.py
```

Hasil yang dihitung:

- accuracy
- balanced accuracy
- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix
- inference latency per image

Hasil disimpan di folder `results/`.

## 7. Prediksi satu gambar

```powershell
python predict.py path\to\tomato_leaf.jpg
```

## 8. Kamera real-time

```powershell
python camera_inference.py
```

## Catatan

File `cnn_train.py` di dalam archive adalah kode CNN Keras lama untuk klasifikasi 10 kelas. Project ini **tidak memakai model tersebut**. Dataset gambarnya saja yang dipakai. Model utama project ini tetap **EfficientNet-B2** sesuai arahan supervisor.
