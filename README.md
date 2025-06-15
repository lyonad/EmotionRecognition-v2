# EmotionRecognition-v2
Versi EmotionRecognition-v1 yang telah dikembangkan

Proyek ini mengimplementasikan sistem deteksi emosi real-time menggunakan kombinasi PCA, K-Nearest Neighbors (KNN), dan algoritma optimasi Pelican Optimization Algorithm (POA). Pertama, citra wajah dari dataset (FER-2013) dimuat, diubah ke skala abu-abu, diubah ukurannya menjadi 48x48 piksel, dan diratakan. Data kemudian dinormalisasi menggunakan `StandardScaler`. Bagian inti proyek adalah penggunaan POA untuk mengoptimalkan **parameter `k` dari KNN (jumlah tetangga terdekat), metode `weights` (bobot), dan `n_components` dari PCA (jumlah komponen utama untuk reduksi dimensi)**, dengan tujuan memaksimalkan akurasi klasifikasi. Setelah parameter optimal ditemukan, model KNN dilatih pada data yang telah direduksi dimensinya oleh PCA. Terakhir, model yang terlatih ini digunakan untuk mendeteksi emosi secara real-time dari tangkapan webcam, menampilkan emosi yang terdeteksi, probabilitas, dan matriks kebingungan.

Cara Memulai Program Utama: 
1. Buka Terminal
2. Masuk ke lokasi folder EmotionDetectionProject, lalu jalankan dengan "python -m streamlit run app.py"
3. Atau Jalankan dengan "python -m streamlit run C:\Users\LyonA\Downloads\EmotionDetectionProject\app.py (lokasi app.py)"

Cara Melatih Model:
1. Buka EmotionRecognition-v2.py
2. Ubah bagian kode ini dengan direktori dataset untuk menjalankan program EmotionRecognition-v2.py: 

train_dir = r'C:\Users\LyonA\Downloads\archive\train'
    
test_dir = r'C:\Users\LyonA\Downloads\archive\test'

3. Jalankan programnya

Dataset: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

Referensi: https://github.com/atulapra/Emotion-detection



----------------------------------------------------------------------------------------------------
Proyek Akhir Artificial Intelligence - Kelompok 11:
1. Lyon Ambrosio Djuanda / 2304130098
2. Rafa Afra Zahirah / 2304130099
3. Naufal Tipasha Denyana / 2304130115