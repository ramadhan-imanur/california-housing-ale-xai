# Modul Komputasi dan Rekayasa Perangkat Lunak: Accumulated Local Effects (ALE)

Dokumentasi teknis implementasi algoritma Accumulated Local Effects (ALE) dan pemodelan regresi harga hunian (California Housing Prices) untuk keperluan riset Mini Paper Versi 1 (V1) Kelompok 3, Mata Kuliah Generatif Kecerdasan Buatan Penulisan Karya Ilmiah, Program Studi S1 Matematika FMIPA Universitas Sebelas Maret (UNS).

---

## 1. Instalasi dan dependensi lingkungan

Proyek ini dibangun di atas Python 3.10+ tanpa dependensi pustaka luar yang tidak stabil. Seluruh kalkulasi kalkulus numerik ALE diimplementasikan secara mandiri dan tervektorisasi menggunakan NumPy dan Pandas.

### Konfigurasi virtual environment dan instalasi package

```bash
# Pindah ke direktori code
cd code

# Buat virtual environment lokal
python3 -m venv .venv

# Aktifkan virtual environment
source .venv/bin/activate

# Pasang dependensi dan instal modul dalam mode editable
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Daftar pustaka utama (`requirements.txt`)

| Pustaka | Fungsi utama |
| :--- | :--- |
| `numpy` | Operasi tensor, differencing lokal vektorisasi, dan akumulasi kumulatif ganda |
| `pandas` | Manipulasi data tabular, kuantil binning, dan one-hot encoding |
| `scipy` | Rutin interpolasi linier dan fungsi kalkulus numerik |
| `scikit-learn` | Pemodelan regresi baseline (`HistGradientBoostingRegressor`, `RandomForestRegressor`, `Pipeline`) |
| `matplotlib` | Rendering visualisasi grafik publikasi ilmiah 300 DPI (`pcolormesh`, plot kontur) |
| `seaborn` | Peta panas korelasi matriks dan penataan visual grafik |
| `pytest` | Pengujian unit otomatis untuk validasi sifat matematis algoritma |

---

## 2. Komponen mesin algoritma (`src/ale_engine.py`)

Modul ini mengimplementasikan formulasi analitis Apley dan Zhu (2020) serta Christoph Molnar (2022) dengan optimasi vektor penuh:
- `ALE1DResult`: Menyimpan `grid_values`, `ale_values` (pemusatan rata-rata nol terhadap distribusi data empiris), `sample_values`, dan `bin_counts`.
- `ALE2DResult`: Menyimpan `grid_x`, `grid_y`, `ale_grid` (matriks interaksi orde-2 murni hasil pemusatan dua arah), dan `cell_counts`.
- `PDP1DResult`: Menyimpan `grid_values`, `pdp_raw`, dan `pdp_centered` (pemusatan berbobot distribusi data empiris yang konsisten dengan ALE).

### Formulasi matematis 1D ALE
$$\hat{f}_{1,ALE}(x_S) = \sum_{k=1}^{k_S(x_S)} \frac{1}{n_S(k)} \sum_{i: x_{i,S} \in [z_{k-1}, z_k)} \left[ \hat{f}(z_k, x_{i,-S}) - \hat{f}(z_{k-1}, x_{i,-S}) \right] - c$$
Interval kuantil $[z_{k-1}, z_k)$ mengisolasi observasi riil tanpa mensintesis kombinasi artifisial di luar distribusi data, dan konstanta pemusatan $c$ dihitung melalui interpolasi linier sehingga $\mathbb{E}[\hat{f}_{1,ALE}(X_S)] = 0$.

### Formulasi matematis 2D ALE (Interaksi murni)
$$\Delta_{j,l}(k, m) = \frac{1}{n_{j,l}(k, m)} \sum_{i \in N_{j,l}(k, m)} \left[ \hat{f}(z_{j,k}, z_{l,m}, x_{i,-(j,l)}) - \hat{f}(z_{j,k-1}, z_{l,m}, x_{i,-(j,l)}) - \hat{f}(z_{j,k}, z_{l,m-1}, x_{i,-(j,l)}) + \hat{f}(z_{j,k-1}, z_{l,m-1}, x_{i,-(j,l)}) \right]$$
Akumulasi kumulatif ganda kemudian dipusatkan menggunakan dekomposisi aditif dua arah berbobot (weighted two-way ANOVA centering):
$$f_{j,l,ALE}(x_j, x_l) = A(k_j, m_l) - \alpha(k_j) - \beta(m_l)$$
sehingga efek marginal orde-1 pada kedua sumbu bernilai nol. Sel kosong atau berobservasi jarang (< 5 sampel) di-masking secara eksplisit untuk mencegah estimasi berisik.

---

## 3. Pipeline pelatihan dan evaluasi (`scripts/train_and_evaluate.py`)

Skrip ini menjalankan alur kerja deterministik dari awal hingga akhir:
1. **Prapemrosesan data**: Memuat 20.640 observasi, mengonversi variabel kategorikal `ocean_proximity` melalui one-hot encoding, dan membagi dataset (80% train, 20% test, `random_state=42`).
2. **Enkapsulasi fitur rasio**: Menghitung fitur turunan domain (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`) di dalam objek `Pipeline` scikit-learn.
3. **Pelatihan model regresi**:
   - `RandomForestRegressor` (100 estimator, `max_depth=16`).
   - `HistGradientBoostingRegressor` (dengan early stopping otomatis).
4. **Pencatatan metrik**: Menghitung koefisien determinasi ($R^2$), Root Mean Squared Error (RMSE), dan Mean Absolute Error (MAE) pada test set.
5. **Kalkulasi dan ekspor grafik**: Seluruh grafik disimpan di `outputs/plots/` dengan resolusi 300 DPI.

Eksekusi pipeline melalui terminal:
```bash
python3 scripts/train_and_evaluate.py
```

Uji otomatis menggunakan pytest:
```bash
pytest
```

---

## 4. Hasil kinerja model baseline

Metrik evaluasi pada data uji independen ($N = 4.128$):

| Model | $R^2$ Score | RMSE (USD) | MAE (USD) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **HistGradientBoostingRegressor** | **0,8465** | **$44.854,01** | **$29.522,20** | Model terbaik (konvergen 212 iterasi) |
| **RandomForestRegressor** | 0,8071 | $50.279,50 | $32.320,03$ | Model pembanding (100 pohon) |

Data hasil evaluasi tersimpan dalam format terstruktur pada `outputs/results_summary.json`.

---

## 5. Ringkasan temuan ALE dan validasi empiris

1. **Efek dominan pendapatan**: Fitur `median_income` memberikan rentang pengaruh marjinal terlebar (~$233.000 USD).
2. **Pemusatan konsisten PDP vs ALE**: Ketika PDP dan ALE dievaluasi pada grid kuantil identik dan dipusatkan terhadap distribusi data empiris, selisih keduanya pada `median_income` berada di kisaran wajar (~$8.800 USD).
3. **Koreksi bias spasial**: Perbedaan kurva ALE vs PDP tampak sangat signifikan pada koordinat spasial (`longitude`, selisih > $110.000 USD) akibat multikolinearitas spasial tinggi ($r = -0{,}92$), membuktikan keunggulan ALE dalam mengabaikan kombinasi bujur-lintang fiktif di lautan.
4. **Interaksi spasial orde-2 murni**: Penerapan pemusatan dua arah (two-way ANOVA) berhasil mengeliminasi efek utama marjinal geografi, mengisolasi efek interaksi murni daratan California dengan masking sel jarang (< 5 sampel) pada rentang ~$108.000 USD.
