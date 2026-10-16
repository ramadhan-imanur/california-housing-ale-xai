# Pemodelan interpretabilitas harga rumah berbasis Accumulated Local Effects (ALE)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-green.svg)](code/tests)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Repositori ini memuat implementasi komputasi dan laporan riset Explainable Artificial Intelligence (XAI) menggunakan metode Accumulated Local Effects (ALE) untuk menginterpretasikan model regresi harga hunian pada dataset kanonik California Housing (sensus 1990 U.S. Census Bureau, Pace & Barry 1997). Riset ini membuktikan secara matematis dan empiris bahwa ALE mampu mengisolasi pengaruh marjinal fitur dan interaksi spasial tanpa bias ekstrapolasi yang lazim ditemukan pada Partial Dependence Plots (PDP).

---

## Ringkasan eksekutif dan temuan utama

Model regresi ensemble seperti Gradient Boosting dan Random Forest menghasilkan akurasi tinggi pada penaksiran harga properti, namun kompleksitas arsitekturnya menyulitkan analisis penetapan nilai aset. Metode interpretasi konvensional seperti Partial Dependence Plots (PDP) mengasumsikan independensi antar-prediktor. Ketika terdapat multikolinearitas tinggi, seperti koordinat bujur terhadap lintang ($r = -0{,}92$) atau total kamar terhadap kamar tidur ($r = 0{,}93$), PDP memaksakan evaluasi pada titik kombinasi fiktif di luar distribusi data empiris (out-of-distribution), misalnya mengevaluasi perumahan California di perairan Samudra Pasifik.

Metode Accumulated Local Effects (ALE) mengeliminasi bias tersebut dengan menghitung selisih prediksi lokal pada interval kuantil data bersyarat yang sempit, mengakumulasikannya sepanjang rentang fitur, dan melakukan pemusatan rata-rata nol terhadap distribusi data empiris.

### 1. Kinerja model regresi ensemble pada data uji (N = 4.128)

| Model regresi | Koefisien determinasi ($R^2$) | RMSE (USD) | MAE (USD) | Status konvergensi |
| :--- | :---: | :---: | :---: | :--- |
| **HistGradientBoostingRegressor** | **0,8465** | **$44.854,01** | **$29.522,20** | Model terbaik (konvergen 212 iterasi dengan early stopping) |
| **RandomForestRegressor** | 0,8071 | $50.279,50 | $32.320,03$ | Model pembanding (100 pohon, kedalaman maksimal 16) |

### 2. Temuan empiris utama analisis ALE

1. **Efek dominan pendapatan riil**: Fitur `median_income` memiliki rentang pengaruh marjinal terlebar terhadap kenaikan nilai hunian, mencapai rentang variasi sekitar $233.000 USD. Evaluasi komparatif pada grid kuantil identik menunjukkan kurva ALE dan PDP yang dipusatkan secara konsisten selaras pada kelas pendapatan menengah, tetapi PDP meremehkan kenaikan harga pada desil pendapatan tinggi hingga selisih lebih dari $100.000 USD akibat perataan marjinal pada data sintetis.
2. **Koreksi distorsi spasial**: Pada fitur bujur geografis (`longitude`), PDP menghasilkan kurva datar yang terdistorsi oleh titik kombinasi fiktif di laut. Sebaliknya, kurva 1D ALE berhasil menangkap lonjakan harga riil di wilayah pesisir barat California dengan selisih efek terhadap PDP melampaui $110.000 USD.
3. **Interaksi spasial dua dimensi murni (2D ALE)**: Penerapan pemusatan dua arah (two-way ANOVA centering) pada interaksi bujur dan lintang berhasil memisahkan efek utama geografis dan menyisakan interaksi spasial murni. Masking eksplisit pada sel observasi jarang (< 5 sampel) menyaring 172 sel daratan berpenghuni, memetakan premium harga di San Francisco Bay Area dan Greater Los Angeles tanpa mengevaluasi koordinat laut.

---

## Metodologi dan formulasi matematis

### 1. Formulasi efek marjinal 1D ALE
Ruang nilai fitur kontinu $X_j$ dipartisi menjadi $K$ interval kuantil: $[z_{0,j}, z_{1,j}], \dots, [z_{K-1,j}, z_{K,j}]$. Efek lokal tak-berpusat diakumulasikan melalui:

$$\hat{g}_{j}(x) = \sum_{k=1}^{k_j(x)} \frac{1}{n_j(k)} \sum_{i: x_{i,j} \in N_j(k)} \left[ \hat{f}(z_{k,j}, x_{i,\setminus j}) - \hat{f}(z_{k-1,j}, x_{i,\setminus j}) \right]$$

dengan $N_j(k) = (z_{k-1,j}, z_{k,j}]$ sebagai interval ke-$k$, $n_j(k)$ adalah jumlah observasi data dalam interval tersebut, dan $x_{i,\setminus j}$ merepresentasikan nilai seluruh prediktor lainnya pada observasi ke-$i$. Pemusatan dilakukan agar nilai ekspektasi efek terhadap distribusi sampel bernilai nol:

$$\hat{f}_{ALE,j}(x) = \hat{g}_{j}(x) - \frac{1}{n} \sum_{i=1}^n \hat{g}_{j}(x_{i,j})$$

### 2. Formulasi interaksi spasial 2D ALE
Interaksi orde dua antara fitur $X_j$ dan $X_l$ dihitung berdasarkan beda hingga orde dua dalam sel kisi $[z_{j,k-1}, z_{j,k}] \times [z_{l,m-1}, z_{l,m}]$:

$$\Delta_{j,l}(k, m) = \frac{1}{n_{j,l}(k, m)} \sum_{i \in N_{j,l}(k, m)} \left[ \hat{f}(z_{j,k}, z_{l,m}, x_{i,\setminus (j,l)}) - \hat{f}(z_{j,k-1}, z_{l,m}, x_{i,\setminus (j,l)}) - \hat{f}(z_{j,k}, z_{l,m-1}, x_{i,\setminus (j,l)}) + \hat{f}(z_{j,k-1}, z_{l,m-1}, x_{i,\setminus (j,l)}) \right]$$

Akumulasi kumulatif ganda kemudian dipusatkan menggunakan proyeksi berselang dua arah (weighted two-way ANOVA centering):

$$f_{j,l,ALE}(x_j, x_l) = A(k_j, m_l) - \alpha(k_j) - \beta(m_l)$$

sehingga rata-rata marjinal baris dan kolom berbobot frekuensi sel bernilai nol, mengisolasi interaksi non-linear murni.

### 3. Protokol prapemrosesan dan rekayasa fitur
Dataset sensus mencakup 20.640 blok sensus di California. Data dibersihkan dan dipartisi dengan rasio 80% data latih (16.512 sampel) dan 20% data uji (4.128 sampel). Fitur rasio domain (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`) diintegrasikan langsung ke dalam objek pipeline scikit-learn. Transformasi ini memastikan bahwa setiap manipulasi nilai fitur induk saat evaluasi ALE secara otomatis memperbarui fitur turunan terkait secara konsisten.

---

## Visualisasi hasil eksperimen

Seluruh visualisasi dihasilkan secara deterministik beresolusi tinggi (300 DPI):
- `01_correlation_matrix.png`: Matriks korelasi Pearson yang memperlihatkan multikolinearitas ekstrem pada koordinat spasial ($r = -0{,}92$) dan fitur demografi perumahan ($r = 0{,}93$).
- `03_ale_1d_all_features.png`: Kurva 1D ALE pada enam prediktor utama beserta rug plot sebaran sampel empiris.
- `04_ale_vs_pdp_median_income.png`: Perbandingan adil antara ALE dan Centered PDP pada grid kuantil identik untuk variabel pendapatan median.
- `05_ale_vs_pdp_longitude.png`: Bukti bias ekstrapolasi PDP akibat korelasi bujur-lintang dibandingkan kestabilan kurva ALE.
- `06_ale_2d_spatial_interaction.png`: Peta interaksi spasial 2D ALE bujur x lintang dengan masking sel jarang (< 5 sampel) yang memetakan klaster harga pesisir California.

---

## Panduan instalasi dan eksekusi

### 1. Persyaratan sistem
- Python versi 3.10 atau yang lebih baru

### 2. Kloning repositori dan instalasi dependensi
```bash
git clone git@github.com:ramadhan-imanur/california-housing-ale-xai.git
cd california-housing-ale-xai/code
pip install -r requirements.txt
pip install -e .
```

### 3. Menjalankan pipeline pelatihan dan evaluasi
Eksekusi skrip pipeline otomatis untuk melatih model, mencatat metrik kinerja, dan mengekspor seluruh grafik publikasi:
```bash
python3 scripts/train_and_evaluate.py
```

### 4. Menjalankan pengujian otomatis unit test
```bash
pytest
```

---

## Konteks akademik dan tim peneliti

Penelitian ini merupakan tugas terstruktur kelompok (Team-Based Project UTS dan UAS) pada mata kuliah Generatif Kecerdasan Buatan Penulisan Karya Ilmiah, Program Studi S1 Matematika, Fakultas Matematika dan Ilmu Pengetahuan Alam (FMIPA), Universitas Sebelas Maret (UNS).

**Dosen Pengampu:** Bapak Nughthoh Arfawi Kurdhi, S.Si., M.Sc., Ph.D.

**Kelompok 3:**
1. Ramadhan Imanur Rochim (M0124015)
2. Oktavian Adam Asmara (M0124043)
3. Reval Aulia Rahma Erlangga (M0124047)
4. Wildan Pratama Faza (M0124049)

---

## Rujukan utama

1. Apley, D. W., & Zhu, J. (2020). Visualizing the effects of predictor variables in black box supervised learning models. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 82(4), 1059-1086.
2. Pace, R. K., & Barry, R. (1997). Sparse spatial autoregressions. *Statistics & Probability Letters*, 33(3), 291-297.
3. Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.). leanpub.com.
4. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.

---

## Lisensi

Proyek ini didistribusikan di bawah lisensi [MIT](LICENSE).
