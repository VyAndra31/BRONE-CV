# BRONE-CV E0

Baseline foto diam: SCRFD 500M KPS (deteksi + lima landmark), alignment 112×112, EdgeFace-XS γ=0.6, embedding L2, cosine similarity, dan penolakan identitas unknown. Seluruh proses berjalan lokal pada CPU. Tidak ada training.

## Persiapan Windows

Jalankan dari `D:\KULIAH\BRONE-CV` di PowerShell. Virtual environment Python 3.12 sudah tersedia di `.venv`. Jika perlu membuat ulang, gunakan Python 3.12 dan `pip install -r requirements.txt`. Repositori EdgeFace resmi ada di `vendor/edgeface`; model SCRFD berasal dari arsip `buffalo_sc.zip` pada [rilis resmi InsightFace v0.7](https://github.com/deepinsight/insightface/releases/tag/v0.7). Checkpoint EdgeFace-XS berasal dari [repositori resmi EdgeFace](https://github.com/otroshi/edgeface). Periksa izin model sebelum distribusi publik.

```powershell
$env:PYTHONPATH='D:\KULIAH\BRONE-CV\src'
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m brone.cli check
.\.venv\Scripts\python.exe -m pytest -q tests
```

`check` memuat kedua model di CPU, menjalankan embedding dummy 112×112, dan melaporkan hash serta dimensi output. Dummy tersebut hanya menguji fungsi, bukan akurasi pengenalan.

## Dataset dan perintah

Salin `data/manifest.example.csv` menjadi `data/manifest.csv`, lalu ganti semua path dan ID dengan foto asli yang telah mendapat persetujuan. Path pada CSV relatif terhadap `data/`. Gunakan satu foto S1 untuk tiap identitas terdaftar. S2 memerlukan foto identitas terdaftar **dan** orang yang tidak ada di gallery (`subject_alias=unknown`). S3 adalah pengujian terpisah. Simpan berkas pada `data/raw/{HP,LAPTOP}/{S1,S2,S3}/{ID}/`.

```powershell
.\.venv\Scripts\python.exe -m brone.cli enroll --manifest data/manifest.csv
.\.venv\Scripts\python.exe -m brone.cli calibrate --manifest data/manifest.csv
.\.venv\Scripts\python.exe -m brone.cli infer data/raw/HP/S3/person_001/foto.jpg
.\.venv\Scripts\python.exe -m brone.cli evaluate --manifest data/manifest.csv
```

Urutan wajib: enrollment dari S1, kalibrasi threshold dari S2, lalu inference/evaluasi S3. Threshold dipilih tepat di atas skor unknown tertinggi di S2. Ini hanya aturan awal empiris; dengan sedikit relawan, jangan menganggapnya jaminan FAR pada populasi lain. Evaluasi tidak menghitung akurasi ketika dataset tidak tersedia.

Pipeline menolak gambar tanpa wajah atau dengan lebih dari satu wajah agar satu label foto tidak tertukar dengan orang lain. `evaluate` mencatat kasus tersebut sebagai status terpisah dalam CSV. Untuk foto banyak wajah, pecah data menjadi gambar satu wajah dengan label yang benar.

## Output

- `features/edgeface_xs/gallery.json`: embedding enrollment dan hash kedua model.
- `configs/threshold.json`: threshold S2, jumlah sampel, metode, dan hash model.
- `data/aligned/{enroll,infer,test}/`: crop 112×112 untuk inspeksi warna dan alignment.
- `results/e0/predictions.csv`: prediksi S3, skor, status deteksi, waktu per modul, dan versi model.
- `results/e0/summary.json`: tingkat keberhasilan deteksi, akurasi known, false accept/reject, serta p50/p95 latency jika sampel tersedia.
- `models/`: bobot SCRFD; `vendor/edgeface/checkpoints/`: bobot EdgeFace-XS.

Semua foto, embedding, bobot, dan hasil lokal dikecualikan dari Git. E1 dan eksperimen berikutnya belum diimplementasikan.
