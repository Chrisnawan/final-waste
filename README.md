# NodeWaste - Streamlit Application

NodeWaste adalah aplikasi web berbasis Streamlit yang dikembangkan untuk melakukan klasifikasi jenis sampah secara interaktif menggunakan model deep learning. Aplikasi ini dibuat agar pengguna dapat menguji model melalui tampilan web yang mudah digunakan, baik secara lokal di komputer maupun secara online melalui Streamlit Community Cloud.

Aplikasi ini menggunakan file model `model.weights.h5` sebagai bobot model deep learning dan file `class_names.txt` sebagai daftar label kelas sampah. Melalui aplikasi ini, pengguna dapat mengunggah gambar sampah, menjalankan proses prediksi, serta melihat hasil klasifikasi berdasarkan kategori sampah yang telah tersedia.

## Struktur Repositori

Repositori ini terdiri dari beberapa file dan folder utama sebagai berikut:

- `app.py` merupakan file utama untuk menjalankan aplikasi Streamlit.
- `requirements.txt` berisi daftar library Python yang dibutuhkan agar aplikasi dapat berjalan.
- `runtime.txt` berisi versi Python yang digunakan pada Streamlit Community Cloud.
- `fix_model.py` merupakan script pembantu untuk konfigurasi atau perbaikan model.
- `model.weights.h5` merupakan file bobot model deep learning yang digunakan aplikasi.
- `class_names.txt` berisi daftar nama kelas atau label sampah.
- `[5] Csv/` berisi data pendukung dalam format CSV.
- `gambar/` berisi aset gambar dan visualisasi pendukung aplikasi.
- `Laporan Komprehensif NodeWaste.pdf` berisi dokumen laporan lengkap proyek NodeWaste.
- `link stremlit claud.txt` berisi link aplikasi yang sudah di-deploy ke Streamlit Community Cloud.

## Cara Menjalankan Aplikasi secara Lokal

Untuk menjalankan aplikasi NodeWaste secara lokal, pengguna perlu mengunduh atau melakukan clone repositori terlebih dahulu melalui terminal atau command prompt. Gunakan perintah berikut:

```bash
git clone https://github.com/username-anda/nama-repositori.git
```

Setelah proses clone selesai, masuk ke dalam folder repositori dengan perintah berikut:

```bash
cd nama-repositori
```

Setelah berada di dalam folder proyek, install seluruh library yang dibutuhkan menggunakan file `requirements.txt` dengan perintah berikut:

```bash
pip install -r requirements.txt
```

Apabila proses instalasi library sudah selesai, aplikasi dapat dijalankan menggunakan perintah Streamlit berikut:

```bash
streamlit run app.py
```

Setelah perintah tersebut dijalankan, Streamlit akan menampilkan alamat lokal pada terminal. Pengguna dapat membuka alamat tersebut melalui browser untuk mulai menggunakan aplikasi NodeWaste.

## Link Aplikasi Online

Aplikasi NodeWaste juga dapat diakses secara online melalui Streamlit Community Cloud pada link berikut:

https://final-waste-3zf8dcxr6apksjoutwunel.streamlit.app/

## Catatan Penting

Pastikan file utama seperti `app.py`, `requirements.txt`, `runtime.txt`, `model.weights.h5`, `class_names.txt`, folder `[5] Csv/`, dan folder `gambar/` tetap berada di dalam repositori. Jika salah satu file utama tersebut tidak tersedia, aplikasi dapat mengalami error saat dijalankan.

Selain itu, pastikan versi Python yang digunakan sudah sesuai dengan file `runtime.txt`, yaitu Python 3.10.11. Hal ini penting agar aplikasi dapat berjalan dengan stabil, terutama saat di-deploy melalui Streamlit Community Cloud.

## Pengembang

Chrisnawan Prastya Atmaja  
Mahasiswa ID: CDCC183D6Y1869
