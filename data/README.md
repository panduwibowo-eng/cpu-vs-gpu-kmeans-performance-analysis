# Data

Folder ini menyimpan dataset mentah yang digunakan dalam eksperimen, dikelompokkan berdasarkan kategori ukuran. **File dataset asli tidak disertakan di repository ini** (beberapa berukuran hingga beberapa GB dan/atau memiliki lisensi redistribusi tersendiri) — gunakan skrip/link pada tiap subfolder untuk mengunduhnya, atau lihat detail sumber lengkap di [`../docs/datasets.md`](../docs/datasets.md).

## Struktur

```
data/
├── small/    # Iris, Seeds, Wine (UCI) — < 1.000 baris
├── medium/   # Wine Quality, Adult, Dry Bean — ribuan-puluhan ribu baris
└── large/    # Covertype, Poker Hand, Sensorless Drive — ratusan ribu baris
```

> Dataset kategori **"Sangat Besar"** (HIGGS, SUSY, Airline Delay — masing-masing ~1 juta baris) sengaja tidak memiliki folder lokal karena diunduh manual dan dimuat langsung dari Google Drive saat eksperimen dijalankan di Google Colab (lihat catatan di [`../docs/datasets.md`](../docs/datasets.md)).

## `data/small/`
| Dataset | Cara memuat |
|---|---|
| Iris | `sklearn.datasets.load_iris()` (built-in, tidak perlu file) |
| Seeds | Unduh dari `https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt` |
| Wine (UCI) | `sklearn.datasets.load_wine()` (built-in, tidak perlu file) |

## `data/medium/`
| Dataset | Cara memuat |
|---|---|
| Wine Quality | Unduh `winequality-red.csv` & `winequality-white.csv` dari UCI, gabungkan |
| Adult | Unduh `adult.data` dari UCI |
| Dry Bean | Unduh `DryBeanDataset.zip` dari UCI |

## `data/large/`
| Dataset | Cara memuat |
|---|---|
| Covertype | `sklearn.datasets.fetch_covtype()` (unduh otomatis) |
| Poker Hand | Unduh `poker-hand-training-true.data` + `poker-hand-testing.data` dari UCI |
| Sensorless Drive | Unduh `Sensorless_drive_diagnosis.txt` dari UCI |

## Catatan Preprocessing

Setiap dataset melalui pipeline yang sama sebelum clustering:
1. Pengecekan missing value dan duplikat.
2. Seleksi/pembuangan kolom non-numerik atau kolom label.
3. Standardisasi fitur dengan `StandardScaler` (mean = 0, std = 1).
4. Konversi ke `cupy` array (untuk jalur GPU) menggunakan `cp.asarray()`.

Detail metodologi lengkap ada di [`../docs/methodology.md`](../docs/methodology.md).
