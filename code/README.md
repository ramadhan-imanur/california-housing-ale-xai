# Modul Komputasi & Rekayasa Perangkat Lunak: Accumulated Local Effects (ALE)

Dokumentasi teknis implementasi algoritma **Accumulated Local Effects (ALE)** dan pemodelan regresi harga rumah (*California Housing Prices*) untuk keperluan riset Mini Paper Versi 1 (V1) Kelompok 3, Mata Kuliah Generatif Kecerdasan Buatan Penulisan Karya Ilmiah, Program Studi S1 Matematika FMIPA Universitas Sebelas Maret (UNS).

---

## 1. Arsitektur Repositori `code/`

Struktur folder dirancang modular memisahkan *engine* algoritma, skrip eksekusi pipeline (*CLI*), lembar kerja eksplorasi (*Jupyter Notebook*), dataset praktikum, unit test, dan artefak luaran:

```text
code/
├── README.md                      # Dokumentasi teknis & panduan reproduksibilitas
├── pyproject.toml                 # Standar konfigurasi packaging modern Python (PEP 621)
├── requirements.txt               # Daftar pustaka dependensi terisolasi & ter-pinned
├── data/                          # Dataset tabular terstruktur
│   └── california_housing_prices.csv
├── src/                           # Library mandiri (reusable Python package)
│   ├── __init__.py                # Package exports
│   └── ale_engine.py              # Algoritma ALE 1D, ALE 2D (two-way ANOVA), dan PDP 1D
├── scripts/                       # Skrip automasi pipeline (CLI)
│   └── train_and_evaluate.py      # Pelatihan model regresi, metrik evaluasi & ekspor grafik
├── notebooks/                     # Eksplorasi interaktif data science
│   └── analisis_ale_california_housing.ipynb
├── tests/                         # Pengujian otomatis perangkat lunak (pytest)
│   └── test_ale_engine.py         # Uji sifat rata-rata nol, pemulihan model aditif, dll.
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

Proyek ini dibangun di atas Python 3.10+ tanpa dependensi pustaka luar yang tidak stabil. Seluruh kalkulasi kalkulus numerik ALE diimplementasikan secara *native* dan tervektorisasi (*vectorized batch evaluation*) menggunakan NumPy dan Pandas.

### A. Konfigurasi Virtual Environment & Instalasi Package

```bash
# Pindah ke direktori code
cd code

# Buat virtual environment lokal
python3 -m venv .venv

# Aktifkan virtual environment
source .venv/bin/activate

# Pasang pustaka dan install modul dalam mode editable (-e .)
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### B. Daftar Pustaka Utama (`requirements.txt`)

| Pustaka | Versi Teruji | Fungsi Utama |
| :--- | :--- | :--- |
| `numpy` | `1.26.4` | Operasi tensor, differencing lokal vektorisasi, dan akumulasi kumulatif ganda |
| `pandas` | `2.1.4` | Manipulasi data tabular, kuantil binning, dan one-hot encoding |
| `scipy` | `1.11.4` | Rutin interpolasi linier dan fungsi kalkulus |
| `scikit-learn` | `1.4.1` | Model baseline (`HistGradientBoostingRegressor`, `RandomForestRegressor`, `Pipeline`) |
| `matplotlib` | `3.11.1` | Rendering visualisasi grafik publikasi ilmiah 300 DPI (`pcolormesh`, plot kontur) |
| `seaborn` | `0.13.2` | Peta panas korelasi matriks dan penataan gaya visual grafik |
| `pytest` | `7.4.4` | Framework otomasi pengujian unit matematis dan struktural |

---

## 3. Komponen Utama

### A. Mesin Algoritma (`src/ale_engine.py`)

Modul ini mengimplementasikan formulasi analitis Apley & Zhu (2020) serta Christoph Molnar (2022) dengan optimasi performa vektor penuh:
- `ALE1DResult`: Menyimpan `grid_values`, `ale_values` (pemusatan rata-rata nol terhadap distribusi data empiris), `sample_values`, dan `bin_counts`.
- `ALE2DResult`: Menyimpan `grid_x`, `grid_y`, `ale_grid` (matriks interaksi orde-2 murni hasil pemusatan dua arah), dan `cell_counts`.
- `PDP1DResult`: Menyimpan `grid_values`, `pdp_raw`, dan `pdp_centered` (pemusatan berbobot distribusi data empiris yang adil/konsisten dengan ALE).

#### 1. Formulasi Matematis 1D ALE:
$$\hat{f}_{1,ALE}(x_S) = \sum_{k=1}^{k_S(x_S)} \frac{1}{n_S(k)} \sum_{i: x_{i,S} \in [z_{k-1}, z_k)} \left[ \hat{f}(z_k, x_{i,-S}) - \hat{f}(z_{k-1}, x_{i,-S}) \right] - c$$
Di mana interval kuantil $[z_{k-1}, z_k)$ mengisolasi observasi riil tanpa mensintesis kombinasi artifisial di luar distribusi data (*out-of-distribution*), dan konstanta pemusatan $c$ dihitung via interpolasi linier sehingga $\mathbb{E}[\hat{f}_{1,ALE}(X_S)] = 0$.

#### 2. Formulasi Matematis 2D ALE (Interaksi Murni):
$$\Delta_{j,l}(k, m) = \frac{1}{n_{j,l}(k, m)} \sum_{i \in N_{j,l}(k, m)} \left[ \hat{f}(z_{j,k}, z_{l,m}, x_{i,-(j,l)}) - \hat{f}(z_{j,k-1}, z_{l,m}, x_{i,-(j,l)}) - \hat{f}(z_{j,k}, z_{l,m-1}, x_{i,-(j,l)}) + \hat{f}(z_{j,k-1}, z_{l,m-1}, x_{i,-(j,l)}) \right]$$
Akumulasi kumulatif ganda kemudian dipusatkan menggunakan dekomposisi aditif dua arah berbobot (*weighted two-way ANOVA centering via alternating projections*):
$$f_{j,l,ALE}(x_j, x_l) = A(k_j, m_l) - \alpha(k_j) - \beta(m_l)$$
sehingga efek marginal orde-1 pada kedua sumbu bernilai nol ($\sum_m n_{k,m} f_{j,l}(k, m) = 0$ dan $\sum_k n_{k,m} f_{j,l}(k, m) = 0$). Sel kosong atau berobservasi jarang ($< 5$ sampel) di-masking secara eksplisit untuk mencegah estimasi berisik.

### B. Pipeline Pelatihan & Evaluasi (`scripts/train_and_evaluate.py`)

Skrip ini menjalankan alur kerja deterministik dari awal hingga akhir (*end-to-end*):
1. **Prapemrosesan Data**: Memuat 20.640 observasi, mengonversi variabel kategorikal `ocean_proximity` melalui *one-hot encoding*, dan membagi dataset (80% train, 20% test, `random_state=42`). Nilai median riil `total_bedrooms` adalah 435.0 (437.0 pada data latih).
2. **Enkapsulasi Fitur Domain**: Menghitung fitur rasio (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`) di dalam objek `Pipeline` scikit-learn agar saat ALE memanipulasi fitur induk, fitur turunan ikut terbarui secara konsisten.
3. **Pelatihan Model Regresi**:
   - `RandomForestRegressor` (100 estimator, `max_depth=16`).
   - `HistGradientBoostingRegressor` (dengan *early stopping* otomatis).
4. **Pencatatan Metrik**: Menghitung koefisien determinasi ($R^2$), *Root Mean Squared Error* (RMSE), dan *Mean Absolute Error* (MAE) pada test set.
5. **Kalkulasi & Ekspor Grafik Reproduktif**: Menggunakan *seeded random generator* (`np.random.default_rng(42)`) sehingga seluruh grafik di `outputs/plots/` memiliki checksum hash deterministik.

Eksekusi skrip melalui terminal:
```bash
python3 scripts/train_and_evaluate.py
```

Uji otomatis menggunakan pytest:
```bash
pytest
```

---

## 4. Hasil Kinerja Model Baseline

Metrik evaluasi pada data uji (*test set* independen, $N = 4.128$):

| Model | $R^2$ Score | RMSE (USD) | MAE (USD) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **HistGradientBoostingRegressor** | **0.8465** | **\$44.854,01** | **\$29.522,20** | Model Terbaik (Konvergen 212 iterasi) |
| **RandomForestRegressor** | 0.8071 | \$50.279,50 | \$32.320,03 | Model Pembanding |

Data hasil evaluasi tersimpan dalam format terstruktur pada `outputs/results_summary.json`.

---

## 5. Ringkasan Temuan ALE & Metodologi

1. **Efek Dominan Pendapatan**: `median_income` memberikan rentang pengaruh marjinal terlebar (~$233.000 USD).
2. **Pemusatan Konsisten PDP vs ALE**: Ketika PDP dan ALE dievaluasi pada grid kuantil yang identik dan dipusatkan terhadap distribusi data empiris, selisih keduanya pada `median_income` berada di kisaran wajar (~$8.800 USD).
3. **Koreksi Bias Spasial**: Perbedaan kurva ALE vs PDP tampak sangat signifikan pada koordinat spasial (`longitude`, selisih > $110.000 USD) akibat multikolinearitas spasial tinggi ($r = -0.92$), membuktikan keunggulan ALE dalam mengabaikan kombinasi bujur-lintang fiktif di lautan atau luar wilayah negara bagian.
4. **Interaksi Spasial Orde-2 Murni**: Penerapan pemusatan dua arah (two-way ANOVA) berhasil mengeliminasi efek utama marjinal geografi, mengisolasi efek interaksi murni daratan California dengan masking sel jarang (< 5 sampel) pada rentang ~$108.000 USD.
