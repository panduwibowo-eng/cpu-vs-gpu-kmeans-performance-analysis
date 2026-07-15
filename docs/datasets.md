# Detail Dataset

Seluruh dataset berasal dari sumber publik (UCI Machine Learning Repository, scikit-learn built-in dataset, atau Kaggle). Dataset dikelompokkan ke dalam 4 kategori ukuran, masing-masing berisi 3 dataset.

## Kategori: Kecil

### Iris
- **Sumber:** `sklearn.datasets.load_iris` (built-in, public domain) — <https://archive.ics.uci.edu/dataset/53/iris>
- **Ukuran:** 150 baris × 4 fitur
- **Kategori kelas:** 3 (setosa, versicolor, virginica) → `n_clusters = 3`

### Seeds
- **Sumber:** UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/236/seeds>
- **Ukuran:** 210 baris × 7 fitur (geometri benih gandum)
- **n_clusters = 3** (3 varietas gandum: Kama, Rosa, Canadian)

### Wine (UCI)
- **Sumber:** UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/109/wine>
- **Ukuran:** 178 baris × 13 fitur (analisis kimia wine Italia)
- **n_clusters = 3** (3 kultivar anggur)

## Kategori: Sedang

### Wine Quality
- **Sumber:** UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/186/wine+quality/>
- **Ukuran:** ~6.497 baris × 11 fitur (red + white wine digabung; setelah `drop_duplicates()` menjadi 5.318 baris)
- **n_clusters = 3** (kualitas wine dikategorikan rendah/sedang/tinggi)

### Adult (Census Income)
- **Sumber:** UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/2/adult>
- **Ukuran:** ~48.842 baris × 6 fitur numerik terpilih (setelah cleaning menjadi 32.334 baris)
- **n_clusters = 3** (segmentasi demografi pendapatan)

### Dry Bean
- **Sumber:** UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset>
- **Ukuran:** 13.611 baris × 16 fitur (morfologi biji kacang)
- **n_clusters = 7** (7 varietas kacang: Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira)

## Kategori: Besar

### Covertype
- **Sumber:** `sklearn.datasets.fetch_covtype` (UCI Covertype Dataset, public domain) — <https://archive.ics.uci.edu/dataset/31/covertype>
- **Ukuran:** 581.012 baris × 54 fitur
- **n_clusters = 7** (sesuai 7 kelas tipe hutan dalam dataset asli)

### Poker Hand
- **Sumber:** UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/158/poker+hand>
- **Ukuran:** 1.025.010 baris × 10 fitur (kartu dalam tangan poker); digunakan sampel 500.000 baris agar waktu eksperimen wajar
- **n_clusters = 10** (10 kelas tangan poker: high card s.d. royal flush)

### Sensorless Drive Diagnosis
- **Sumber:** UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis>
- **Ukuran:** 58.509 baris × 48 fitur (sinyal motor listrik, deteksi kerusakan)
- **n_clusters = 11** (11 kondisi komponen motor)

## Kategori: Sangat Besar

> Ketiga dataset berikut diunduh manual dan disimpan di Google Drive karena ukuran filenya yang besar (ratusan MB – beberapa GB), lalu dimuat via `drive.mount()`.

### HIGGS
- **Sumber:** UCI Machine Learning Repository — <https://archive.ics.uci.edu/ml/datasets/HIGGS> (~7,5 GB)
- **Ukuran yang digunakan:** 1.000.000 baris × 28 fitur (kolom label dihapus)
- **n_clusters = 7** (klasifikasi fisika partikel: berbagai mode peluruhan)

### SUSY
- **Sumber:** UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/279/susy> (~2,4 GB)
- **Ukuran yang digunakan:** 1.000.000 baris × 18 fitur (sinyal fisika partikel supersimetri)
- **n_clusters = 5** (segmentasi pola sinyal partikel)

### Airline Delay
- **Sumber:** Kaggle — 2015 Flight Delays and Cancellations — <https://www.kaggle.com/datasets/usdot/flight-delays> (~565 MB)
- **Ukuran yang digunakan:** 1.000.000 baris (setelah cleaning menjadi 944.644 baris) × 5 fitur numerik
- **n_clusters = 5** (segmentasi pola keterlambatan penerbangan)

## Tabel Referensi Cepat

| Dataset | Kategori | Sumber |
|---|---|---|
| Iris | Kecil | https://archive.ics.uci.edu/dataset/53/iris |
| Seeds | Kecil | https://archive.ics.uci.edu/dataset/236/seeds |
| Wine (UCI) | Kecil | https://archive.ics.uci.edu/dataset/109/wine |
| Wine Quality | Sedang | https://archive.ics.uci.edu/dataset/186/wine+quality |
| Adult | Sedang | https://archive.ics.uci.edu/dataset/2/adult |
| Dry Bean | Sedang | https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
| Covertype | Besar | https://archive.ics.uci.edu/dataset/31/covertype |
| Poker Hand | Besar | https://archive.ics.uci.edu/dataset/158/poker+hand |
| Sensorless Drive | Besar | https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis |
| HIGGS | Sangat Besar | https://archive.ics.uci.edu/dataset/280/higgs |
| SUSY | Sangat Besar | https://archive.ics.uci.edu/dataset/279/susy |
| Airline Delay | Sangat Besar | https://www.kaggle.com/datasets/usdot/flight-delays |

Lihat juga [`../data/README.md`](../data/README.md) untuk panduan penempatan file dataset di struktur repository ini.
