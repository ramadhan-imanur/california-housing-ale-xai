# Accumulated Local Effects (ALE) for Interpretable House Price Modeling

Implementasi Explainable Artificial Intelligence (XAI) berbasis metode Accumulated Local Effects (ALE) dan komparasi dengan Partial Dependence Plots (PDP) pada dataset California Housing (sumber sensus Pace & Barry, 1997).

Proyek ini dikembangkan oleh Kelompok 3 Program Studi S1 Matematika FMIPA Universitas Sebelas Maret (UNS) untuk mendemonstrasikan bagaimana metode ALE mengisolasi pengaruh marginal fitur numerik dan interaksi spasial tanpa terdistorsi oleh multikolinearitas.

---

## Ringkasan Eksekutif

Model black-box seperti Random Forest dan Gradient Boosting sering digunakan untuk memprediksi harga properti karena akurasi prediksinya yang tinggi. Namun, teknik interpretasi global standar seperti Partial Dependence Plots (PDP) mengasumsikan independensi antar fitur. Ketika fitur berkorelasi kuat (misalnya pendapatan rata-rata terhadap jumlah kamar, atau bujur terhadap lintang), PDP mengevaluasi prediksi pada kombinasi data yang tidak realistis di luar distribusi data asli (*out-of-distribution*).

Metode Accumulated Local Effects (ALE) mengatasi kelemahan ini dengan:
1. Menghitung perbedaan prediksi lokal dalam jendela interval bersyarat sempit.
2. Mengakumulasikan turunan lokal tersebut di sepanjang rentang fitur.
3. Melakukan pemusatan (*mean-centering*) sehingga nilai rata-rata efek ALE bernilai nol.

Eksperimen pada dataset California Housing menunjukkan bahwa ALE memberikan estimasi efek marginal yang stabil tanpa melebih-lebihkan dampak fitur pada ekor distribusi, serta mampu memetakan interaksi spasial 2D (koordinat bujur dan lintang) yang merefleksikan harga properti di wilayah pesisir California secara realistis.

---

## Struktur Direktori

```text
.
├── LICENSE                          # Lisensi publikasi kode (MIT License)
├── README.md                        # Dokumentasi utama repositori
├── code/
│   ├── README.md                    # Dokumentasi teknis modul komputasi
│   ├── requirements.txt             # Daftar pustaka dependensi Python
│   ├── data/
│   │   └── california_housing_prices.csv  # Dataset acuan kanonik
│   ├── src/
│   │   ├── __init__.py              # Inisialisasi package Python
│   │   └── ale_engine.py            # Kalkulasi native NumPy untuk ALE 1D, ALE 2D, dan PDP 1D
│   ├── scripts/
│   │   └── train_and_evaluate.py    # Pipeline pelatihan model dan ekspor grafik
│   ├── notebooks/
│   │   └── analisis_ale_california_housing.ipynb # Eksplorasi interaktif data science
│   └── outputs/
│       ├── plots/                   # Visualisasi ilmiah 300 DPI
│       │   ├── 01_correlation_matrix.png
│       │   ├── 03_ale_1d_all_features.png
│       │   ├── 04_ale_vs_pdp_median_income.png
│       │   ├── 05_ale_vs_pdp_longitude.png
│       │   └── 06_ale_2d_spatial_interaction.png
│       └── results_summary.json     # Rekapitulasi metrik numerik model
```

---

## Kinerja Model Regresi

Eksperimen menggunakan tiga arsitektur model regresi dengan pembagian 80% data latih dan 20% data uji:

| Model | R² Score | RMSE (USD) | MAE (USD) |
| :--- | :---: | :---: | :---: |
| **XGBoost Regressor** | **0.8354** | **46.852** | **30.985** |
| **Random Forest Regressor** | 0.8122 | 50.043 | 32.812 |
| **Linear Regression (Baseline)** | 0.6062 | 72.410 | 52.880 |

Model XGBoost dan Random Forest dijadikan model dasar untuk komparasi kurva interpretasi ALE dan PDP.

---

## Cara Menjalankan

### 1. Kloning Repositori
```bash
git clone git@github.com:ramadhan-imanur/california-housing-ale-xai.git
cd california-housing-ale-xai/code
```

### 2. Konfigurasi Environment & Dependensi
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Menjalankan Pipeline Komputasi
Untuk melatih model, menghitung kurva ALE, dan memperbarui seluruh grafik visualisasi:
```bash
python3 scripts/train_and_evaluate.py
```

### 4. Menjalankan Notebook Interaktif
```bash
jupyter lab notebooks/analisis_ale_california_housing.ipynb
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
