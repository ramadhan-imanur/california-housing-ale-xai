# Modul Komputasi & Rekayasa Perangkat Lunak: Accumulated Local Effects (ALE)

Dokumentasi teknis implementasi algoritma **Accumulated Local Effects (ALE)** dan pemodelan regresi harga rumah (*California Housing Prices*) untuk keperluan riset Mini Paper Versi 1 (V1) Kelompok 3, Mata Kuliah Generatif Kecerdasan Buatan Penulisan Karya Ilmiah, Program Studi S1 Matematika FMIPA Universitas Sebelas Maret (UNS).

---

## 1. Arsitektur Repositori `code/`

Struktur folder dirancang modular memisahkan *engine* algoritma, skrip eksekusi pipeline (*CLI*), lembar kerja eksplorasi (*Jupyter Notebook*), dataset praktikum, dan artefak luaran:

```
code/
├── README.md                      # Dokumentasi teknis & panduan reproduksibilitas
├── requirements.txt               # Daftar pustaka dependensi terisolasi
├── data/                          # Dataset tabular terstruktur
│   └── california_housing_prices.csv
├── src/                           # Library mandiri (reusable Python package)
│   ├── __init__.py                # Package exports
│   └── ale_engine.py              # Algoritma ALE 1D, ALE 2D, dan PDP 1D native NumPy/Pandas
├── scripts/                       # Skrip automasi pipeline (CLI)
│   └── train_and_evaluate.py      # Pelatihan model regresi, metrik evaluasi & ekspor grafik
├── notebooks/                     # Eksplorasi interaktif data science
│   └── analisis_ale_california_housing.ipynb
└── outputs/                       # Seluruh artefak luaran komputasi
    ├── plots/                     # Visualisasi ilmiah beresolusi tinggi (300 DPI)
    │   ├── 01_correlation_matrix.png
    │   ├── 03_ale_1d_all_features.png
    │   ├── 04_ale_vs_pdp_median_income.png
    │   ├── 05_ale_vs_pdp_longitude.png
    │   └── 06_ale_2d_spatial_interaction.png
    └── results_summary.json       # Rekapitulasi metrik numerik model (R2, RMSE, MAE)
```

---

## 2. Instalasi & Dependensi

Proyek ini dibangun di atas Python 3.10+ tanpa dependensi pustaka luar yang tidak stabil. Seluruh kalkulasi kalkulus numerik ALE diimplementasikan secara *native* menggunakan NumPy dan Pandas.

### A. Konfigurasi Virtual Environment

```bash
# Pindah ke direktori code
cd "/media/ramadhan/0C6A-1ABD/University/Generative AI/Tugas 1/code"

# Buat virtual environment lokal
python3 -m venv .venv

# Aktifkan virtual environment
source .venv/bin/activate

# Pasang pustaka yang diperlukan
pip install --upgrade pip
pip install -r requirements.txt
```

### B. Daftar Pustaka Utama (`requirements.txt`)

| Pustaka | Versi Minimal | Fungsi Utama |
| :--- | :--- | :--- |
| `numpy` | `>=1.26.0` | Operasi tensor, differencing lokal, dan komputasi kumulatif |
| `pandas` | `>=2.1.0` | Manipulasi data tabular, kuantil binning, dan one-hot encoding |
| `scipy` | `>=1.11.0` | Rutin interpolasi linier dan fungsi statistik |
| `scikit-learn` | `>=1.4.0` | Model baseline (`RandomForestRegressor`, `HistGradientBoostingRegressor`) |
| `matplotlib` | `>=3.8.0` | Rendering visualisasi grafik publikasi ilmiah 300 DPI |
| `seaborn` | `>=0.13.0` | Peta panas korelasi matriks dan penataan gaya visual grafik |

---

## 3. Komponen Utama

### A. Mesin Algoritma (`src/ale_engine.py`)

Modul ini mengimplementasikan formulasi analitis Apley & Zhu (2020) serta Christoph Molnar (2022). Tiga kelas kontainer keluaran berorientasi data (`dataclass`):
- `ALE1DResult`: Menyimpan `grid_values`, `ale_values` (pemusatan rata-rata nol), `sample_values`, dan `bin_counts`.
- `ALE2DResult`: Menyimpan `grid_x`, `grid_y`, `ale_grid` (matriks $K \times K$ interaksi orde-2), dan `cell_counts`.
- `PDP1DResult`: Menyimpan `grid_values`, `pdp_raw`, dan `pdp_centered` untuk pembanding.

#### Formulasi 1D ALE:
$$\hat{f}_{1,ALE}(x_S) = \sum_{k=1}^{k_S(x_S)} \frac{1}{n_S(k)} \sum_{i: x_{i,S} \in N_S(k)} \left[ \hat{f}(z_k, x_{i,-S}) - \hat{f}(z_{k-1}, x_{i,-S}) \right] - c$$
Di mana interval kuantil $N_S(k) = (z_{k-1}, z_k]$ mengisolasi observasi riil tanpa mensintesis kombinasi artifisial di luar distribusi data (*out-of-distribution data points*).

### B. Pipeline Pelatihan & Evaluasi (`scripts/train_and_evaluate.py`)

Skrip ini menjalankan alur kerja deterministik dari awal hingga akhir (*end-to-end*):
1. **Prapemrosesan Data**: Memuat 20.640 observasi, mengimputasi *missing values* pada `total_bedrooms` dengan median (535), mengonversi variabel kategorikal `ocean_proximity` melalui *one-hot encoding*, dan membagi dataset (80% train, 20% test, `random_state=42`).
2. **Pelatihan Model Regresi**:
   - `RandomForestRegressor` (100 estimator, `max_depth=16`).
   - `HistGradientBoostingRegressor` (150 iterasi).
3. **Pencatatan Metrik**: Menghitung koefisien determinasi ($R^2$), *Root Mean Squared Error* (RMSE), dan *Mean Absolute Error* (MAE).
4. **Kalkulasi & Ekspor Grafik**: Menghasilkan 5 visualisasi 300 DPI ke `outputs/plots/` dan ringkasan metrik ke `outputs/results_summary.json`.

Eksekusi skrip melalui terminal:
```bash
python3 scripts/train_and_evaluate.py
```

### C. Analisis Eksploratif Interaktif (`notebooks/analisis_ale_california_housing.ipynb`)

Notebook Jupyter interaktif yang membedah setiap tahapan analisis:
- Eksplorasi multikolinearitas (misal: korelasi spasial $r=-0.92$ antara `longitude` dan `latitude`, korelasi struktural $r=0.93$ antara `total_rooms` dan `total_bedrooms`).
- Pembandingan empiris ALE vs. PDP pada `median_income` (membuktikan PDP mengalami bias marjinal hingga \$100.000 USD pada kelompok pendapatan tinggi).
- Pemetaan kontur spasial 2D ALE pada daratan California (hanya mengevaluasi 172 sel terisi dari 361 kisi grid).

Jalankan notebook dengan:
```bash
jupyter notebook notebooks/analisis_ale_california_housing.ipynb
```

---

## 4. Hasil Kinerja Model Baseline

Metrik evaluasi pada data uji (*test set* independen, $N = 4.128$):

| Model | $R^2$ Score | RMSE (USD) | MAE (USD) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **HistGradientBoostingRegressor** | **0.8286** | **\$47.398,71** | **\$31.730,67** | Model Terbaik (Baseline Mini Paper) |
| **RandomForestRegressor** | 0.8141 | \$49.356,32 | \$32.089,22 | Model Pembanding |

Data hasil evaluasi tersimpan dalam format terstruktur pada [results_summary.json](file:///media/ramadhan/0C6A-1ABD/University/Generative%20AI/Tugas%201/code/outputs/results_summary.json).

---

## 5. Ringkasan Temuan ALE untuk Mini Paper

1. **Efek Dominan Pendapatan**: `median_income` memberikan rentang pengaruh marjinal terlebar ($-60.000$ USD hingga $+222.000$ USD).
2. **Koreksi Bias Multikolinearitas**: Kurva PDP menghasilkan estimasi semu karena memaksakan evaluasi kombinasi ekstrem yang tidak pernah terjadi di dunia nyata, sedangkan ALE mempertahankan estimasi murni bersyarat pada data lokal.
3. **Interaksi Spasial Orde-2**: Kombinasi koordinat pesisir pantai California (Bay Area dan Los Angeles) menunjukkan efek interaksi positif yang signifikan, sementara wilayah pedalaman memiliki efek interaksi netral/negatif.
