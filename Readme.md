# 🧭 WOOFI ML

Sebuah proyek machine learning yang memberikan rekomendasi tempat wisata di Indonesia secara personal, berdasarkan preferensi pengguna dan deskripsi konten wisata. Sistem ini dirancang untuk mendukung pengalaman eksplorasi wisata yang lebih cerdas, relevan, dan menyenangkan.
link web : [WOOFI](https://www.woofi.web.id/)

---

## Dataset: `Tempat_Wisata.csv`

- **Sumber Data**: [Google Maps](https://maps.google.com)
- **Metode Pengambilan**: Manual scraping berdasarkan tempat wisata populer dari berbagai wilayah seperti **Sumatera Selatan**, **Lampung**, **Jawa Barat**, dan lainnya.

## Dataset: `test.json`

- **Sumber Data**: Data pengguna untuk pengujian sistem rekomendasi tempat wisata
- **Deskripsi**: Data simulasi pengguna yang digunakan untuk menguji sistem rekomendasi. Masing-masing entri merepresentasikan satu user lengkap dengan informasi demografi, minat, dan riwayat pencarian tempat wisata.

### Fitur yang Dikumpulkan:

- **Nama Tempat Wisata**
- **Rating**
- **Kategori**
- **Provinsi**
- **Link Google Maps**
- **Deskripsi (Bahasa Inggris)**
- **Foto**

---

## 🤖 Teknologi & Machine Learning

### Model:

- **Supervised Classification Model**  
  Untuk memprediksi tempat wisata berdasarkan gender, usia, dan minat pengguna.

- **Content-Based Filtering**  
  Menggunakan TF-IDF + cosine similarity dari deskripsi tempat wisata untuk mencari kemiripan dengan preferensi pengguna.

### Fitur Pendukung:

- **Pencarian Multibahasa** dengan Google Translate API.
- **Rekomendasi Personal** untuk user baru & lama:
  - User baru: berdasarkan profil (gender, umur, interest)
  - User lama: berdasarkan histori pencarian dan ketertarikan
- **Model gabungan** (supervised + content-based) untuk hasil rekomendasi yang lebih akurat.

---

## 🛠️ Fitur Aplikasi

- Rekomendasi destinasi wisata berdasarkan minat pengguna
- Pencarian dan penyaringan berdasarkan provinsi atau kategori

---

## 📦 Instalasi Library

### Dari requirements.txt

pip install -r requirements.txt
