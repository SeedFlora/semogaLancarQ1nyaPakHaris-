# semogaLancarQ1nyaPakHaris-

Kode analisis tambahan untuk naskah IJOST *"Frozen DINOv3 and Multilingual-E5 Embeddings with
CatBoost for Multimodal Triage of Citizen Environmental Reports"*. Kode ini menjawab catatan
reviewer ("kurangan dari DINO-EVA2"), yang meminta analisis **per-sampel** pada test set
(n = 9.266):

1. **Kalibrasi**: ECE (15 bin), Brier, NLL, dan reliability diagram, **dengan dan tanpa PGS**,
   plus temperature scaling sebagai pembanding.
2. **Uji berpasangan antar-encoder**: McNemar eksak dan paired bootstrap (Δakurasi, Δmacro-F1)
   untuk DINOv3-L, EVA-02-L, dan DINOv2-L (masing-masing dengan mE5-L), dengan koreksi Holm.
3. **Selective prediction**: kurva akurasi–cakupan, AURC, dan AUROC ketidakpastian terhadap
   kesalahan, untuk max-probability dan dekomposisi PGS (entropi total, aleatorik, epistemik/MI).

Tidak ada model yang dilatih ulang. Skrip hanya memuat cache embedding dan checkpoint CatBoost
yang sudah ada, lalu menjalankan inferensi (±0,05 ms per sampel).

## Dokumen revisi (`docs/`)

| Berkas | Isi |
|---|---|
| `docs/Ringkasan_Revisi_DINO-EVA2.pdf` | Ringkasan untuk pembimbing (6 halaman): catatan yang diminta, kendala, rancangan replikasi, hasil, dampak pada klaim, daftar perubahan per bagian, dan hal yang perlu dicek |
| `docs/SmartCitty_IJOST_Rev_1_tracked.pdf` | Naskah revisi dengan tracked changes dan komentar terlihat |
| `docs/SmartCitty_IJOST_Rev_1_clean.pdf` | Naskah revisi dengan semua perubahan diterima (37 halaman) |

## Alur kerja

```
[RunPod]  export_per_sample_predictions.py  → per_sample_predictions/*.csv.gz + manifest.json
[lokal ]  analyze_uncertainty.py            → results/ (tabel, gambar, teks naskah, claims_check.md)
[lokal ]  fill_manuscript.py                → naskah .docx dengan placeholder ⟦…⟧ terisi
```

Format data antar-tahap, definisi statistik, dan daftar placeholder ada di
[`CONTRACT.md`](CONTRACT.md).

### 1. Ekspor prediksi per-sampel (di RunPod, tempat `artifacts/` berada)

```bash
pip install -r requirements.txt            # atau cukup: catboost numpy pandas scikit-learn
python export_per_sample_predictions.py --artifacts /workspace/smartCityReport/artifacts \
       --out per_sample_predictions
```

Struktur yang dibaca (sama dengan notebook 06/07/08):
`artifacts/embeddings/{image,text}/<encoder>.npy`, `artifacts/splits/{train,val,test}.csv`, dan
`artifacts/models/checkpoints/<img>__<txt>__{cb,pgs}.cbm`.

Ringkasan di akhir otomatis membandingkan hasil dengan angka naskah (Tabel 4: 0.7996/0.7684,
0.7914/0.7600, 0.8116/0.7793, 0.8073–0.8074/0.7747; 130 beda top-1; `prob_std` ≈ 0.00339).
**Kalau muncul blok `WARNING`, checkpoint yang tersimpan bukan yang dipakai di naskah. Berhenti
dan periksa dulu sebelum lanjut.**

### 2. Analisis (lokal)

```bash
python analyze_uncertainty.py --preds per_sample_predictions --out results \
       --manuscript path/ke/SmartCitty_IJOST_Rev_1.docx
```

Baca `results/claims_check.md` lebih dulu. Isinya verdict SUPPORTED / QUALIFIED / CONTRADICTED
untuk tiap klaim naskah yang diuji, beserta kalimat naskah yang perlu diubah bila klaim itu tidak
didukung data.

### 3. Isi naskah

```bash
python fill_manuscript.py --docx SmartCitty_IJOST_Rev_1.docx --results results \
       --out SmartCitty_IJOST_Rev_2.docx
```

Skrip mengisi ke-84 placeholder `⟦…⟧` dan menukar Gambar 5–6. Jika masih ada yang kosong,
skrip keluar dengan kode 1. File input tidak pernah ditimpa.

## Replikasi (data per-sampel dibuat ulang)

Cache embedding dan checkpoint run asli di RunPod sudah terhapus, dan snapshot 61.773 pasangan
tidak bisa diambil lagi. Karena itu Subbab 3.4 naskah diisi dari **run replikasi**: pipeline yang
sama, dijalankan ulang pada mirror Drive yang tersisa. Mirror ini memuat 86.888 record mentah dan
menghasilkan 37.503 pasangan bersih dari tahun 2023, dengan split 26.251 / 5.626 / 5.626 (seed 42).
Semua skripnya ada di [`replication/`](replication/):

| Tahap | Skrip | Isi |
|---|---|---|
| data | `prepare_data.py` | logika notebook 02 (pemetaan 9 kelas, dedup narasi) dan notebook 03 (split 70/15/15, seed 42); baris yang gambarnya hilang permanen dibuang sebelum split |
| gambar | `extract_from_zips.py`, `download_images.py` | ambil gambar dari zip ekspor Google Drive; sisanya diunduh pelan (resumable) |
| model | `download_models.py` | bobot DINOv3-L (port timm), DINOv2-L, EVA-02-L, mE5-L (safetensors) |
| embedding | `extract_embeddings.py` | encoder beku persis seperti notebook 04/05/08. Processor DINOv3 dicek bit-identik dengan transformers. Perilaku bawaan transformers 4.x dipertahankan (bobot fp32, processor PIL) |
| head | `train_catboost.py` | CatBoost baseline dan PGS dengan parameter notebook 08, untuk 3 pasangan encoder |
| semua | `run_pipeline.py` | menjalankan semua tahap berurutan, bisa dilanjutkan dari tahap terakhir; lalu `export_*`, `analyze_*`, dan `summarize_run.py` |

```bash
pip install -r requirements.txt -r requirements-gpu.txt   # + torch sesuai versi CUDA
export CRM_SRC=/path/ke/smartCityReport/src               # paket crm dari repo skripsi
export REPL_METADATA=/path/ke/metadata.csv                # metadata mirror Drive
python replication/extract_from_zips.py "/path/ke/Thesis Data-*.zip"   # opsional
cd replication && python run_pipeline.py
```

Run ini memakai GPU (RTX 3070 Ti Laptop, 8 GB). Ekstraksi embedding sengaja dibatasi (2 worker DataLoader), dan
semua unduhan memakai satu koneksi agar laptop tetap stabil.

### Hasil replikasi (test n = 5.626)

| Analisis | Hasil |
|---|---|
| Kalibrasi, checkpoint PGS | ECE argmax 0,0271 vs PGS-averaged 0,0379. Selisih berpasangan +0,0107 [0,0054; 0,0163], jadi PGS **memperburuk** kalibrasi; Brier dan NLL juga naik |
| Temperature scaling (T = 0,90) | ECE 0,0204 (tidak signifikan); Brier dan NLL turun signifikan |
| DINOv3-L vs EVA-02-L vs DINOv2-L | tidak ada selisih signifikan setelah Holm (DINOv3 vs EVA-02: p = 0,257) |
| PGS vs argmax, checkpoint sama | −0,36 pp akurasi, McNemar p = 0,0119 (PGS menurunkan akurasi) |
| Selisih naif (checkpoint PGS vs baseline) | +0,27 pp macro-F1, p = 0,168 (tidak signifikan) |
| AUROC deteksi kesalahan | 1 − max p (tanpa PGS) 0,8041; MI epistemik PGS 0,6447 |
| Penundaan 20% laporan paling ragu | akurasi laporan yang dirutekan otomatis 0,8102 → 0,8809 |

`replication/finalize_text.py` menghitung uji berpasangan untuk ΔECE, ΔBrier dan ΔNLL, lalu menyusun teks akhir naskah. Pilihan kata (misalnya "memperburuk" atau "tidak signifikan") ditentukan dari interval kepercayaan hasil uji tersebut.

## Docker

```powershell
docker build -t q1-uncertainty .
docker run --rm q1-uncertainty                                  # jalankan seluruh tes (data sintetis)

# analisis + pengisian naskah dari folder saat ini (PowerShell)
docker run --rm -v "${PWD}:/work" q1-uncertainty `
  python analyze_uncertainty.py --preds /work/per_sample_predictions --out /work/results
docker run --rm -v "${PWD}:/work" q1-uncertainty `
  python fill_manuscript.py --docx /work/SmartCitty_IJOST_Rev_1.docx --results /work/results --out /work/SmartCitty_IJOST_Rev_2.docx
```

Image yang sama juga bisa menjalankan ekspor, asal folder `artifacts/` di-mount
(`-v /workspace/smartCityReport/artifacts:/artifacts` lalu `--artifacts /artifacts`).

## Hasil yang sudah bisa dihitung sekarang (`mcnemar_bounds.py`)

Uji McNemar hanya bergantung pada pasangan diskordan b dan c. Akurasi yang dilaporkan menentukan
b − c secara pasti, sedangkan b + c dibatasi oleh jumlah kesalahan. Karena itu, p-value kasus
terburuk bisa dihitung tanpa data per-sampel:

| Perbandingan (angka dari naskah) | p terburuk | Kesimpulan |
|---|---|---|
| Fusi vs. teks saja, head PGS (Tabel 3) | ≤ 0,003 | signifikan |
| Fusi vs. teks saja, tanpa PGS (Tabel 3) | ≤ 0,031 | signifikan |
| PGS-averaged vs. argmax, checkpoint PGS (≤ 130 beda top-1) | ≤ 7,6 × 10⁻⁴ | PGS signifikan **menurunkan** akurasi |
| Checkpoint baseline, dan perbandingan naif | ≤ 0,22–0,25 | belum konklusif, perlu data per-sampel |

```bash
python mcnemar_bounds.py
```

## Tes

```bash
python -m pytest tests -q      # 95 tes: ekspor, analisis, pengisian, dan rantai ujung-ke-ujung
```

Semua tes memakai data sintetis (`make_synthetic_fixture.py`, `tests/synthetic_per_sample.py`).
Rumus statistik diverifikasi terhadap implementasi independen (statsmodels, sklearn, dan loop naif).

## Data dan privasi

Repositori ini **hanya berisi kode**. Dataset CRM Jakarta (foto dan narasi warga, belum
dianonimkan), cache embedding, checkpoint `.cbm`, hasil prediksi, dan naskah `.docx` sengaja tidak
disertakan dan diblokir oleh `.gitignore`.

## Isi repositori

| File | Fungsi |
|---|---|
| `export_per_sample_predictions.py` | Tahap 1: ekspor prediksi per-sampel (argmax + PGS M = 30, linear & log-linear pooling, MI/entropi) dan pemeriksaan reproduksi angka naskah |
| `analyze_uncertainty.py` | Tahap 2: ECE/Brier/NLL, temperature scaling, McNemar, paired bootstrap, Holm, AUROC, AURC, akurasi–cakupan, gambar, teks naskah, `claims_check.md` |
| `fill_manuscript.py` | Tahap 3: isi placeholder dan gambar pada DOCX (tracked changes tetap utuh) |
| `mcnemar_bounds.py` | Batas p McNemar kasus-terburuk dari akurasi agregat |
| `make_synthetic_fixture.py` | Membuat `artifacts/` tiruan (embedding + checkpoint CatBoost kecil) untuk pengujian |
| `CONTRACT.md` | Kontrak antarmuka antar-tahap |
| `tests/` | Uji otomatis |
| `replication/` | Run ulang pipeline (data → gambar → embedding → CatBoost) untuk data per-sampel |
| `docs/` | PDF ringkasan revisi dan naskah (versi tracked dan versi bersih) |
| `Dockerfile`, `requirements.txt`, `requirements-gpu.txt` | Lingkungan yang dapat direproduksi (Python 3.12, catboost 1.2.10; GPU untuk embedding) |
