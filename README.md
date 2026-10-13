# Accumulated Local Effects (ALE) for Interpretable House Price Modeling

Implementasi Explainable Artificial Intelligence (XAI) berbasis metode Accumulated Local Effects (ALE) dan komparasi dengan Partial Dependence Plots (PDP) pada dataset California Housing (sumber sensus Pace & Barry, 1997).

Proyek ini dikembangkan oleh Kelompok 3 Program Studi S1 Matematika FMIPA Universitas Sebelas Maret (UNS) untuk mendemonstrasikan bagaimana metode ALE mengisolasi pengaruh marginal fitur numerik dan interaksi spasial tanpa terdistorsi oleh multikolinearitas.

---

## Ringkasan Eksekutif

Model machine learning ensemble seperti Random Forest dan Gradient Boosting sering digunakan untuk memprediksi harga properti karena akurasi prediksinya yang tinggi. Namun, teknik interpretasi global standar seperti Partial Dependence Plots (PDP) mengasumsikan independensi antar fitur. Ketika fitur berkorelasi kuat (misalnya bujur terhadap lintang dengan $r = -0.92$, atau jumlah kamar terhadap kamar tidur dengan $r = 0.93$), PDP mengevaluasi prediksi pada kombinasi data yang tidak realistis di luar distribusi data asli (*out-of-distribution*).

Metode Accumulated Local Effects (ALE) mengatasi kelemahan ini dengan:
1. Menghitung perbedaan prediksi lokal dalam jendela interval kuantil bersyarat sempit.
2. Mengakumulasikan efek lokal tersebut di sepanjang rentang fitur secara invektorisasi.
3. Melakukan pemusatan (*mean-centering*) terhadap distribusi sampel empiris sehingga $\mathbb{E}[\hat{f}_{ALE}(X)] = 0$.
4. Menerapkan dekomposisi aditif dua arah berbobot (*two-way ANOVA centering*) untuk 2D ALE guna mengisolasi interaksi spasial murni dan mem-masking sel data kosong/jarang.

---

## Struktur Direktori

```text
.
├── LICENSE                          # Lisensi publikasi kode (MIT License)
├── README.md                        # Dokumentasi utama repositori
├── code/
│   ├── README.md                    # Dokumentasi teknis modul komputasi & reproduksibilitas
│   ├── pyproject.toml               # Konfigurasi instalasi modul Python (PEP 621)
│   ├── requirements.txt             # Daftar dependensi ter-pinned
│   ├── data/
│   │   └── california_housing_prices.csv  # Dataset sensus California 1990 kanonik
│   ├── src/
│   │   ├── __init__.py              # Inisialisasi package Python
│   │   └── ale_engine.py            # Kalkulasi vectorized NumPy untuk ALE 1D, ALE 2D, dan PDP 1D
│   ├── scripts/
│   │   └── train_and_evaluate.py    # Pipeline pelatihan model, evaluasi, dan ekspor grafik
│   ├── notebooks/
│   │   └── analisis_ale_california_housing.ipynb # Eksplorasi interaktif data science
│   ├── tests/
│   │   └── test_ale_engine.py       # Unit test otomasi (pytest)
│   └── outputs/
│       ├── plots/                   # Visualisasi ilmiah deterministik 300 DPI
│       │   ├── 01_correlation_matrix.png
│       │   ├── 03_ale_1d_all_features.png
│       │   ├── 04_ale_vs_pdp_median_income.png
│       │   ├── 05_ale_vs_pdp_longitude.png
│       │   └── 06_ale_2d_spatial_interaction.png
│       └── results_summary.json     # Rekapitulasi metrik numerik model
```

---

## Kinerja Model Regresi

Eksperimen menggunakan model ensemble yang teruji secara langsung pada test set independen ($N = 4.128$, 20% data) dengan rekayasa fitur rasio terenkapsulasi dalam pipeline:

| Model | $R^2$ Score | RMSE (USD) | MAE (USD) | Catatan Pelatihan |
| :--- | :---: | :---: | :---: | :--- |
| **HistGradientBoostingRegressor** | **0.8465** | **44.854** | **29.522** | Model Utama (Konvergen 212 iterasi dengan early stopping) |
| **RandomForestRegressor** | 0.8071 | 50.280 | 32.320 | Model Pembanding (100 trees, max_depth=16) |

---

## Cara Menjalankan

### 1. Kloning Repositori
```bash
git clone https://github.com/ramadhan-imanur/california-housing-ale-xai.git
cd california-housing-ale-xai/code
```

### 2. Konfigurasi Environment & Dependensi
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3. Menjalankan Pipeline Komputasi
Untuk melatih model, menghitung kurva ALE, dan meregenerasi seluruh visualisasi deterministik:
```bash
python3 scripts/train_and_evaluate.py
```

### 4. Menjalankan Unit Tests
```bash
pytest
```

### 5. Menjalankan Notebook Interaktif
```bash
jupyter notebook notebooks/analisis_ale_california_housing.ipynb
```

---

## Rujukan Utama

1. Apley, D. W., & Zhu, J. (2020). *Visualizing the effects of predictor variables in black box supervised learning models*. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 82(4), 1059-1086.
2. Pace, R. K., & Barry, R. (1997). *Sparse spatial autoregressions*. Statistics & Probability Letters, 33(3), 291-297.
3. Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.). Munchen: leanpub.com.
4. Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. The Annals of Statistics, 29(5), 1189-1232.

---

## Lisensi

Proyek ini didistribusikan di bawah lisensi [MIT License](LICENSE).
