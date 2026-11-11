# Klasifikasi Sampah Rumah Tangga Berbasis Computer Vision

**Perbandingan MobileNetV3 dan EfficientNet-B0**

Proyek akhir mata kuliah **Computer Vision**, Program Studi S-1 Ilmu Komputer, Fakultas Ilmu Komputer dan Teknologi Informasi, Universitas Sumatera Utara (2026). **Kelompok 2.**

Proyek ini membandingkan dua arsitektur CNN ringan, **MobileNetV3** dan **EfficientNet-B0** (transfer learning), untuk mengklasifikasikan citra sampah rumah tangga ke tiga kategori: **plastik**, **kertas/karton**, dan **logam**. Perbandingan dilakukan pada dataset dan protokol pengujian yang sama, dengan metrik accuracy, precision/recall/F1 per kelas (macro F1 sebagai pembanding utama), confusion matrix, dan waktu inferensi.

---

## Status Proyek

> Terakhir diperbarui: 11 November 2026

| Tahap | Rencana (dari proposal) | Status |
|---|---|---|
| 1. Proposal dan pemilihan dataset | Minggu 1 (2-8 Nov) | Bagian Dataset selesai disusun, penggabungan ke proposal kelompok menunggu konfirmasi |
| 2. Persiapan data | Minggu 2 (9-15 Nov) | **Sedang berjalan.** Skrip persiapan data selesai, kode augmentasi belum |
| 3. Pelatihan dan tuning model | Minggu 3 (16-22 Nov) | Belum dimulai |
| 4. Evaluasi dan analisis perbandingan | Minggu 4 (23-29 Nov) | Belum dimulai |
| 5. Laporan akhir dan bahan presentasi | Minggu 5 (30 Nov-4 Des) | Belum dimulai |
| 6. Presentasi akhir | Minggu 6 (7-18 Des) | Belum dimulai |

### Rincian pekerjaan

**Tahap 1: Proposal dan dataset**
- [x] Memilih dataset: TrashNet, dipetakan ke 3 kelas
- [x] Mengunduh dan memverifikasi dataset (1.889 citra terpilih)
- [x] Draf Bagian 5 proposal (Dataset): sumber, jumlah data, contoh gambar, split, pra-pemrosesan, augmentasi
- [ ] Bagian 5 digabung ke proposal kelompok dan dikumpulkan (Checkpoint 2)

**Tahap 2: Persiapan data**
- [x] Skrip persiapan data (`prepare_dataset.py`), sudah diuji pada beberapa struktur folder
- [x] Split latih/validasi/uji berbasis kelompok (`data/splits.csv`)
- [x] Bobot kelas untuk loss (`data/class_weights.json`)
- [x] Spesifikasi augmentasi dan pratinjau (`docs/figures/`)
- [ ] Menyepakati framework (PyTorch atau Keras) dengan Anggota 2 dan 3
- [ ] Kode DataLoader dan augmentasi sesuai framework
- [ ] Membagikan `splits.csv` agar semua model memakai split yang sama

**Tahap 3-6: Belum dimulai**
- [ ] Baseline dan tuning MobileNetV3
- [ ] Baseline dan tuning EfficientNet-B0 dengan protokol yang sama
- [ ] Evaluasi pada data uji: metrik, confusion matrix, waktu inferensi
- [ ] Analisis perbandingan hasil
- [ ] Laporan akhir dan presentasi

---

## Dataset

**TrashNet** (Thung & Yang, 2016), foto sampah di atas papan putih, dipotret dengan iPhone. Versi yang dipakai adalah `dataset-resized` (512x384 piksel, JPEG, RGB).

- Repositori: https://github.com/garythung/trashnet (lisensi MIT)
- Berkas: https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip

Dataset **tidak diunggah ke repositori ini**. Jalankan `prepare_dataset.py` untuk mengunduhnya otomatis.

### Seleksi kelas

| Kelas TrashNet | Kelas proyek | Jumlah | Persentase |
|---|---|---|---|
| plastic | Plastik | 482 | 25,5% |
| paper + cardboard | Kertas/Karton | 997 (594 + 403) | 52,8% |
| metal | Logam | 410 | 21,7% |
| **Total dipakai** | | **1.889** | 100% |
| glass, trash | tidak dipakai | 638 (501 + 137) | |

Rasio kelas terbesar terhadap terkecil adalah 2,43 : 1.

### Pembagian data (seed 42)

| Kelas | Latih | Validasi | Uji | Total |
|---|---|---|---|---|
| Plastik | 337 | 72 | 73 | 482 |
| Kertas/Karton | 698 | 150 | 149 | 997 |
| Logam | 286 | 62 | 62 | 410 |
| **Total** | **1.321** | **284** | **284** | **1.889** |

### Temuan pemeriksaan data

- Semua berkas terbaca dan berformat RGB. Tidak ada duplikat byte-identik pada tiga kelas terpilih.
- Sebagian citra adalah **foto objek yang sama dari sudut sedikit berbeda** (near-duplicate). Deteksi otomatis (dHash 256-bit, jarak Hamming <= 12, dalam kelas yang sama) menemukan 10 kelompok yang berisi 25 citra (sekitar 1,3%). Jumlah sebenarnya bisa lebih banyak karena dataset tidak menyertakan identitas objek.
- Split acak akan memisahkan sebagian pasangan kembar ke data latih dan data uji (kebocoran data, akurasi terlalu optimis). Karena itu split dibuat **stratified dan berbasis kelompok** sehingga kembaran selalu berada di subset yang sama.

Gambar pendukung ada di `docs/figures/`:

| Gambar | Isi |
|---|---|
| `Gambar_1_Contoh_Citra_Per_Kelas.png` | 15 contoh citra, 5 per kelas |
| `Gambar_2_Distribusi_Kelas_dan_Split.png` | Distribusi kelas dan pembagian latih/validasi/uji |
| `Gambar_3_Pratinjau_Augmentasi.png` | Contoh hasil augmentasi |

---

## Pra-pemrosesan dan augmentasi (rencana)

**Pra-pemrosesan** (sama untuk kedua model): resize 224x224 (bilinear, tanpa crop), skala piksel ke [0, 1], normalisasi mean/std ImageNet (0,485; 0,456; 0,406) / (0,229; 0,224; 0,225). Jika framework sudah memasukkan normalisasi di dalam model (misalnya Keras), langkah ini menyesuaikan. Ketidakseimbangan kelas ditangani dengan bobot kelas pada loss (`data/class_weights.json`), bukan oversampling.

**Augmentasi** (hanya data latih, on-the-fly, identik untuk kedua model):

| Teknik | Parameter |
|---|---|
| Flip horizontal | p = 0,5 |
| Flip vertikal | p = 0,5 |
| Rotasi | +-30 derajat, area kosong diisi putih |
| Zoom dan translasi | skala 0,85-1,15x, geser hingga 10% |
| Color jitter | brightness +-30%, contrast +-30%, saturation +-20%, hue +-0,02 |
| Gaussian blur | kernel 3, sigma 0,1-1,0, p = 0,2 |

Augmentasi tidak mengubah latar menjadi ramai, sehingga ketahanan model pada foto berlatar kompleks tetap perlu diuji dan dicatat sebagai keterbatasan.

---

## Cara menjalankan

Persyaratan: Python 3.10 atau lebih baru.

```bash
# 1. Buat dan aktifkan virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

# 2. Instal pustaka
pip install -r requirements.txt

# 3a. Jika BELUM punya dataset: unduh otomatis
python prepare_dataset.py --out dataset_final

# 3b. Jika SUDAH punya dataset (folder berisi cardboard, glass, metal, paper, plastic, trash)
python prepare_dataset.py --data-dir "path/ke/folder_dataset" --out dataset_final
```

Hasil di folder `dataset_final/`:

```
dataset_final/
  train/  plastik/  kertas_karton/  logam/
  val/    plastik/  kertas_karton/  logam/
  test/   plastik/  kertas_karton/  logam/
  splits.csv
  class_weights.json
```

Skrip akan menampilkan PERINGATAN jika jumlah citra per kelas berbeda dari TrashNet resmi. Jika itu muncul, angka pada tabel di atas perlu dihitung ulang.

> Agar semua anggota memakai split yang sama, gunakan `data/splits.csv` dari repositori ini sebagai acuan bersama.

---

## Struktur repositori

```
.
├── README.md
├── requirements.txt
├── prepare_dataset.py        # persiapan data: seleksi kelas, deteksi kembaran, split, salin file
├── data/
│   ├── splits.csv            # daftar citra, kelas asal, label, grup, dan subset (train/val/test)
│   └── class_weights.json    # bobot kelas dari data latih
└── docs/
    └── figures/              # gambar untuk proposal dan laporan
```

Folder dataset, virtual environment, dan bobot model tidak diunggah (lihat `.gitignore`).

---

## Anggota Kelompok 2

Fikri Tamimi, Charissa Haya Qanita, Desi Maria Elita Silalahi, Diva Syakirahana, Muhammaad Sulthan Zaki Nasution

| Peran | Tanggung jawab |
|---|---|
| Anggota 1: Data Specialist | Dataset, pra-pemrosesan, split, augmentasi |
| Anggota 2: ML Engineer, Model A | MobileNetV3 dengan transfer learning |
| Anggota 3: ML Engineer, Model B | EfficientNet-B0 dengan transfer learning, protokol yang sama dengan Model A |
| Anggota 4 | Task, metrik evaluasi, dan timeline |
| Anggota 5 | Judul, latar belakang, tujuan, dan pembagian tugas |

## Referensi

- Thung, G., & Yang, M. (2016). *Classification of Trash for Recyclability Status* (CS229 Project Report). Stanford University. https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf
- Thung, G. (2017). *TrashNet* [Dataset]. GitHub. https://github.com/garythung/trashnet