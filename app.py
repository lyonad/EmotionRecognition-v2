import streamlit as st
import cv2
import numpy as np
import time
import pickle
from PIL import Image
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Fungsi untuk memuat model dari file
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        knn = model_data['knn']
        pca = model_data['pca']
        scaler = model_data['scaler']
        emotions = model_data['emotions']
        
        # Load face cascade
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(face_cascade_path):
            st.warning("Mengunduh face cascade...")
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml" 
            urllib.request.urlretrieve(url, "haarcascade_frontalface_default.xml")
            face_cascade_path = "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        return knn, pca, scaler, emotions, face_cascade
    except Exception as e:
        st.error(f"Error saat memuat model: {str(e)}")
        return None, None, None, None, None

# Fungsi untuk memproses wajah
def preprocess_face(face_img, target_size=(48, 48)):
    face_img = cv2.resize(face_img, target_size)
    face_vector = face_img.flatten()
    return face_vector

# Fungsi untuk memprediksi emosi
def predict_emotion(face_img, knn, pca, scaler):
    face_vector = preprocess_face(face_img)
    face_vector = face_vector.reshape(1, -1)
    face_scaled = scaler.transform(face_vector)
    face_pca = pca.transform(face_scaled)
    emotion_idx = knn.predict(face_pca)[0]
    emotion_probs = knn.predict_proba(face_pca)[0]
    return emotion_idx, emotion_probs

# Fungsi untuk memproses frame
def process_frame(frame, knn, pca, scaler, emotions, face_cascade, show_probabilities=True):
    emotion_colors = {
        0: (0, 0, 255),    # Marah: Merah
        1: (0, 255, 0),    # Jijik: Hijau
        2: (255, 0, 255),  # Takut: Ungu
        3: (0, 255, 255),  # Senang: Kuning
        4: (255, 255, 255),# Netral: Putih
        5: (255, 0, 0),    # Sedih: Biru
        6: (255, 255, 0),  # Terkejut: Cyan
    }
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    detected_faces = []
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        
        try:
            emotion_idx, emotion_probs = predict_emotion(face_roi, knn, pca, scaler)
            emotion = emotions[emotion_idx]
            probability = emotion_probs[emotion_idx]
            color = emotion_colors.get(emotion_idx, (255, 255, 255))
            
            # Gambar kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Tampilkan emosi dan probabilitas
            text = f"{emotion}: {probability:.2f}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Tampilkan grafik probabilitas
            if show_probabilities:
                bar_width = 100
                bar_height = 15
                offset_y = y + h + 20
                
                for i, prob in enumerate(emotion_probs):
                    prob_width = int(bar_width * prob)
                    emotion_color = emotion_colors.get(i, (200, 200, 200))
                    cv2.rectangle(frame, (x, offset_y), (x + prob_width, offset_y + bar_height), emotion_color, -1)
                    emotion_text = f"{emotions[i]}: {prob:.2f}"
                    cv2.putText(frame, emotion_text, (x + bar_width + 10, offset_y + bar_height), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 1)
                    offset_y += bar_height + 5
            
            detected_faces.append({
                "x": x, "y": y, "width": w, "height": h,
                "emotion": emotion, "probability": probability,
                "all_probabilities": emotion_probs
            })
            
        except Exception as e:
            st.warning(f"Error dalam prediksi emosi: {str(e)}")
    
    return frame, detected_faces

# Fungsi untuk membuat chart emosi
def create_emotion_chart(emotions, probabilities):
    fig = px.bar(
        x=emotions, 
        y=probabilities,
        labels={'x': 'Emosi', 'y': 'Probabilitas'},
        title='Distribusi Probabilitas Emosi',
        color=emotions,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(
        xaxis_title="Emosi",
        yaxis_title="Probabilitas",
        showlegend=False,
        yaxis=dict(range=[0, 1])
    )
    return fig

# Fungsi utama Streamlit
def main():
    st.set_page_config(
        page_title="Sistem Deteksi Emosi Real-time", 
        layout="wide",
        page_icon="😊"
    )
    
    # CSS untuk styling
    st.markdown("""
    <style>
    /* Perbaikan tampilan tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 15px;
        border-radius: 8px 8px 0 0;
        font-size: 14px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4b7bec;
        color: white !important;
    }
    
    .stTabs [aria-selected="true"] div div {
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #d6e4ff;
    }
    
    /* Perbaikan layout header */
    .header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header h1 {
        font-size: 2rem !important;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .header p {
        font-size: 1rem !important;
        opacity: 0.9;
    }
    
    .feature-section {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .feature-box {
        flex: 1;
        min-width: 200px;
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .feature-box h3 {
        margin-top: 0;
        color: #3498db;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.1rem;
    }
    
    .feature-icon {
        font-size: 1.3rem;
    }
    
    .stats-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .stat-box {
        flex: 1;
        min-width: 150px;
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3498db;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #7f8c8d;
    }
    
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1.2rem;
        color: #7f8c8d;
        font-size: 0.85rem;
        border-top: 1px solid #ecf0f1;
    }
    
    /* Responsif untuk mobile */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 12px;
        }
        
        .header h1 {
            font-size: 1.6rem !important;
        }
        
        .header p {
            font-size: 0.9rem !important;
        }
        
        .stat-value {
            font-size: 1.3rem !important;
        }
        
        .stat-box, .feature-box {
            min-width: 100%;
        }
        
        .feature-section, .stats-container {
            flex-direction: column;
            gap: 0.8rem;
        }
    }
    
    .performance-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    
    .performance-good {
        background-color: #d4edda;
        color: #155724;
    }
    
    .performance-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .performance-poor {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .history-item {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-bottom: 0.5rem;
        border-left: 4px solid #4b7bec;
    }
    
    .history-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .history-emotion {
        font-weight: bold;
        font-size: 1.1rem;
        color: #2c3e50;
    }
    
    .delete-btn {
        background-color: #ff6b6b;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header">
        <h1>😊 Sistem Deteksi Emosi Real-time</h1>
        <p>Deteksi emosi wajah dari gambar, video, atau webcam menggunakan model machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistik
    st.markdown("""
    <div class="stats-container">
        <div class="stat-box">
            <div class="stat-value">7</div>
            <div class="stat-label">Jenis Emosi</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">Real-time</div>
            <div class="stat-label">Deteksi Webcam</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">POA</div>
            <div class="stat-label">Optimasi Model</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">AI</div>
            <div class="stat-label">Computer Vision</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Fitur
    st.markdown("""
    <div class="feature-section">
        <div class="feature-box">
            <h3><span class="feature-icon">📷</span> Deteksi Gambar</h3>
            <p>Unggah gambar wajah dan dapatkan analisis emosi secara instan dengan visualisasi probabilitas.</p>
        </div>
        <div class="feature-box">
            <h3><span class="feature-icon">🎥</span> Deteksi Video</h3>
            <p>Analisis emosi secara real-time melalui webcam dengan interval penyimpanan hasil.</p>
        </div>
        <div class="feature-box">
            <h3><span class="feature-icon">📊</span> Analisis Historis</h3>
            <p>Tinjau hasil deteksi sebelumnya dengan statistik dan visualisasi data.</p>
        </div>
        <div class="feature-box">
            <h3><span class="feature-icon">⚙️</span> Optimasi POA</h3>
            <p>Model teroptimasi dengan algoritma Pelican untuk akurasi deteksi emosi.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar untuk unggah model dan pengaturan
    with st.sidebar:
        st.header("⚙️ Konfigurasi Model")
        uploaded_model = st.file_uploader(
            "Unggah Model (.pkl)", 
            type=["pkl"],
            help="Unggah file model yang telah dilatih (format .pkl)"
        )
        
        # Inisialisasi state untuk model
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
            st.session_state.knn = None
            st.session_state.pca = None
            st.session_state.scaler = None
            st.session_state.emotions = None
            st.session_state.face_cascade = None
            st.session_state.detection_history = []
            st.session_state.show_probabilities = True
            st.session_state.real_time_settings = {
                'frame_skip': 1,
                'resolution': 0.8,
                'camera_index': 0,
                'detection_interval': 10  # Default 10 detik
            }
        
        # Pengaturan deteksi
        st.subheader("⚙️ Pengaturan Deteksi")
        st.session_state.show_probabilities = st.checkbox(
            "Tampilkan Probabilitas", 
            value=True,
            help="Tampilkan grafik probabilitas untuk semua emosi"
        )
        
        # Pengaturan real-time
        st.subheader("⚙️ Pengaturan Real-time")
        st.session_state.real_time_settings['frame_skip'] = st.slider(
            "Skip Frame", 
            1, 5, 2,
            help="Melewatkan beberapa frame untuk meningkatkan performa"
        )
        st.session_state.real_time_settings['resolution'] = st.slider(
            "Resolusi Webcam", 
            0.5, 1.0, 0.8,
            help="Mengurangi resolusi untuk meningkatkan performa"
        )
        st.session_state.real_time_settings['camera_index'] = st.selectbox(
            "Pilih Kamera",
            options=[0, 1, 2],
            index=0,
            help="Pilih perangkat kamera yang akan digunakan"
        )
        
        # Pengaturan interval deteksi
        st.subheader("⏱ Pengaturan Interval Deteksi")
        st.session_state.real_time_settings['detection_interval'] = st.slider(
            "Interval Penyimpanan (detik)", 
            5, 60, 10,
            help="Interval waktu untuk menyimpan hasil deteksi dalam mode real-time"
        )
        
        # Tombol reset
        if st.button("🔄 Reset Model dan Data"):
            st.session_state.model_loaded = False
            st.session_state.knn = None
            st.session_state.pca = None
            st.session_state.scaler = None
            st.session_state.emotions = None
            st.session_state.face_cascade = None
            st.session_state.detection_history = []
            st.success("Model dan data deteksi telah direset!")
        
        # Informasi model
        if st.session_state.model_loaded:
            st.subheader("ℹ️ Informasi Model")
            st.info(f"**Emosi yang Didukung:** {', '.join(st.session_state.emotions)}")
            st.info(f"**Algoritma:** KNN (K-Nearest Neighbors)")
            st.info(f"**Jumlah Komponen PCA:** {st.session_state.pca.n_components}")
            st.info(f"**Jumlah Tetangga:** {st.session_state.knn.n_neighbors}")
            st.info(f"**Bobot:** {st.session_state.knn.weights}")
        
        # Panduan
        st.subheader("❓ Panduan Penggunaan")
        st.info("""
        1. Unggah model (.pkl) melalui menu di atas
        2. Pilih mode deteksi (Gambar atau Real-time)
        3. Untuk deteksi gambar: unggah gambar wajah
        4. Untuk deteksi real-time: aktifkan webcam
        """)
        
        # Sejarah deteksi
        if st.session_state.detection_history:
            st.subheader("🕒 Riwayat Deteksi")
            history_container = st.container()
            with history_container:
                for i, item in enumerate(st.session_state.detection_history[:5]):
                    if 'image' in item and 'emotion' in item:
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.image(item['image'], width=60)
                        with cols[1]:
                            st.markdown(f"**{item['emotion']}**")
                            st.caption(f"{item['probability']:.2f} | {item['time']}")
    
    # Proses file model yang diunggah
    if uploaded_model is not None:
        try:
            # Simpan file model sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_file.write(uploaded_model.getvalue())
                model_path = tmp_file.name
            
            # Muat model
            knn, pca, scaler, emotions, face_cascade = load_model(model_path)
            
            if knn is not None:
                st.session_state.knn = knn
                st.session_state.pca = pca
                st.session_state.scaler = scaler
                st.session_state.emotions = emotions
                st.session_state.face_cascade = face_cascade
                st.session_state.model_loaded = True
                st.sidebar.success("Model berhasil dimuat!")
                
                # Tampilkan info model
                with st.expander("ℹ️ Detail Model yang Dimuat", expanded=True):
                    st.write(f"**Emosi yang Didukung:** {', '.join(emotions)}")
                    st.write(f"**Jumlah Komponen PCA:** {pca.n_components}")
                    st.write(f"**Jumlah Tetangga (k):** {knn.n_neighbors}")
                    st.write(f"**Bobot:** {knn.weights}")
                    
                    # Visualisasi komponen PCA
                    if hasattr(pca, 'explained_variance_ratio_'):
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.bar(range(1, len(pca.explained_variance_ratio_[:10])+1), 
                               pca.explained_variance_ratio_[:10])
                        ax.set_xlabel('Komponen PCA')
                        ax.set_ylabel('Proporsi Varians')
                        ax.set_title('10 Komponen PCA Teratas')
                        st.pyplot(fig)
            else:
                st.error("Gagal memuat model. Pastikan file model valid.")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
    
    # Pilihan mode dengan tab yang diperbaiki
    tab1, tab2, tab3 = st.tabs([
        "📷 Gambar", 
        "🎥 Real-time", 
        "📊 Historis"
    ])
    
    with tab1:
        st.header("📷 Deteksi Emosi dari Gambar")
        
        if not st.session_state.model_loaded:
            st.warning("Silakan unggah model terlebih dahulu di sidebar")
        else:
            uploaded_file = st.file_uploader(
                "Unggah gambar wajah", 
                type=["jpg", "jpeg", "png"],
                key="image_uploader"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
                
                if st.button("Deteksi Emosi", key="detect_image", use_container_width=True):
                    frame = np.array(image)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    with st.spinner("Menganalisis emosi..."):
                        try:
                            processed_frame, detected_faces = process_frame(
                                frame.copy(), 
                                st.session_state.knn, 
                                st.session_state.pca, 
                                st.session_state.scaler, 
                                st.session_state.emotions, 
                                st.session_state.face_cascade,
                                st.session_state.show_probabilities
                            )
                            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            processed_image = Image.fromarray(processed_frame)
                            
                            with col2:
                                st.image(processed_image, caption="Hasil Deteksi", use_column_width=True)
                                st.success("Deteksi selesai!")
                            
                            # Simpan ke riwayat
                            if detected_faces:
                                for face in detected_faces:
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.session_state.detection_history.insert(0, {
                                        "type": "image",
                                        "time": timestamp,
                                        "emotion": face["emotion"],
                                        "probability": face["probability"],
                                        "all_probabilities": face["all_probabilities"],
                                        "image": image.copy()
                                    })
                            
                            # Tampilkan hasil analisis
                            if detected_faces:
                                st.subheader("📊 Hasil Analisis Emosi")
                                for i, face in enumerate(detected_faces):
                                    st.markdown(f"### Wajah #{i+1}")
                                    cols = st.columns(2)
                                    with cols[0]:
                                        st.markdown(f"**Emosi Dominan:** {face['emotion']}")
                                        st.markdown(f"**Probabilitas:** {face['probability']:.4f}")
                                    with cols[1]:
                                        fig = create_emotion_chart(
                                            st.session_state.emotions, 
                                            face['all_probabilities']
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            # Opsi unduh hasil
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                processed_image.save(tmp_file, format='JPEG')
                                tmp_file_path = tmp_file.name
                            
                            with open(tmp_file_path, "rb") as file:
                                st.download_button(
                                    label="📥 Unduh Hasil Deteksi",
                                    data=file,
                                    file_name="hasil_deteksi_emosi.jpg",
                                    mime="image/jpeg",
                                    use_container_width=True
                                )
                            
                        except Exception as e:
                            st.error(f"Terjadi kesalahan: {str(e)}")
    
    with tab2:
        st.header("🎥 Deteksi Emosi Real-time")
        
        if not st.session_state.model_loaded:
            st.warning("Silakan unggah model terlebih dahulu di sidebar")
        else:
            interval = st.session_state.real_time_settings['detection_interval']
            st.info(f"Tekan tombol 'Mulai Deteksi' di bawah untuk memulai deteksi menggunakan webcam")
            st.warning(f"Hasil deteksi akan disimpan setiap {interval} detik")
            
            run = st.checkbox("Mulai Deteksi", key="run_detection")
            stop_button = st.button("Berhenti", key="stop_detection")
            
            FRAME_WINDOW = st.image([], width=640)
            status_text = st.empty()
            stats_text = st.empty()
            snapshot_placeholder = st.empty()
            
            # Inisialisasi webcam
            cap = None
            if run:
                cap = cv2.VideoCapture(st.session_state.real_time_settings['camera_index'])
                if not cap.isOpened():
                    st.error("Tidak dapat mengakses webcam")
                    run = False
            
            frame_count = 0
            start_time = time.time()
            last_save_time = time.time()  # Waktu terakhir penyimpanan
            fps = 0
            frame_skip_counter = 0
            last_snapshot = None
            
            while run and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    status_text.error("Gagal mengambil frame dari webcam")
                    break
                
                # Skip frame sesuai pengaturan
                frame_skip_counter += 1
                if frame_skip_counter < st.session_state.real_time_settings['frame_skip']:
                    continue
                frame_skip_counter = 0
                
                # Resize untuk performa
                scale = st.session_state.real_time_settings['resolution']
                if scale < 1.0:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
                # Proses frame
                try:
                    processed_frame, detected_faces = process_frame(
                        frame.copy(), 
                        st.session_state.knn, 
                        st.session_state.pca, 
                        st.session_state.scaler, 
                        st.session_state.emotions, 
                        st.session_state.face_cascade,
                        st.session_state.show_probabilities
                    )
                    
                    # Hitung FPS
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        fps = frame_count / elapsed_time
                    
                    # Tampilkan FPS
                    fps_text = f"FPS: {fps:.2f}"
                    cv2.putText(processed_frame, fps_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Konversi warna untuk Streamlit
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(processed_frame_rgb, channels="RGB")
                    
                    # Periksa apakah sudah waktunya menyimpan deteksi
                    current_time = time.time()
                    detection_interval = st.session_state.real_time_settings['detection_interval']
                    time_since_last_save = current_time - last_save_time
                    
                    # Simpan deteksi setiap interval tertentu
                    if detected_faces and time_since_last_save >= detection_interval:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Cari wajah dengan probabilitas tertinggi
                        main_face = max(detected_faces, key=lambda x: x['probability'])
                        
                        # Simpan snapshot
                        snapshot_img = Image.fromarray(processed_frame_rgb.copy())
                        
                        # Simpan ke riwayat
                        st.session_state.detection_history.insert(0, {
                            "type": "video",
                            "time": timestamp,
                            "emotion": main_face['emotion'],
                            "probability": main_face['probability'],
                            "all_probabilities": main_face['all_probabilities'],
                            "image": snapshot_img.copy(),
                            "interval": detection_interval
                        })
                        
                        # Update waktu terakhir penyimpanan
                        last_save_time = current_time
                        
                        # Simpan sebagai snapshot terakhir
                        last_snapshot = {
                            "frame": processed_frame_rgb.copy(),
                            "timestamp": timestamp,
                            "detections": detected_faces
                        }
                    
                    # Update status
                    fps_status = ""
                    if fps > 15:
                        fps_status = "<span class='performance-badge performance-good'>Baik</span>"
                    elif fps > 8:
                        fps_status = "<span class='performance-badge performance-medium'>Sedang</span>"
                    else:
                        fps_status = "<span class='performance-badge performance-poor'>Lambat</span>"
                    
                    # Hitung waktu sampai penyimpanan berikutnya
                    next_save = max(0, detection_interval - (current_time - last_save_time))
                    
                    status_text.markdown(f"""
                    **Status Deteksi:**  
                    - FPS: {fps:.2f} {fps_status}  
                    - Wajah Terdeteksi: {len(detected_faces)}  
                    - Penyimpanan berikutnya: {next_save:.1f} detik
                    - Waktu: {datetime.now().strftime("%H:%M:%S")}
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    status_text.error(f"Error dalam pemrosesan frame: {str(e)}")
                    break
                
                # Reset penghitung FPS setiap detik
                if elapsed_time > 1:
                    frame_count = 0
                    start_time = time.time()
            
            if cap is not None:
                cap.release()
            
            if stop_button:
                run = False
                status_text.success("Deteksi dihentikan")
                FRAME_WINDOW.image([])
            
            # Tombol untuk menyimpan snapshot terakhir
            if last_snapshot is not None:
                snapshot_placeholder.subheader("💾 Snapshot Terakhir")
                snapshot_col1, snapshot_col2 = st.columns(2)
                
                with snapshot_col1:
                    st.image(last_snapshot["frame"], caption=f"Snapshot - {last_snapshot['timestamp']}")
                
                with snapshot_col2:
                    if last_snapshot["detections"]:
                        st.markdown("**Deteksi Wajah:**")
                        for i, face in enumerate(last_snapshot["detections"]):
                            st.markdown(f"**Wajah #{i+1}:** {face['emotion']} ({face['probability']:.2f})")
                
                # Opsi unduh snapshot
                snapshot_img = Image.fromarray(last_snapshot["frame"])
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    snapshot_img.save(tmp_file, format='JPEG')
                    tmp_file_path = tmp_file.name
                
                with open(tmp_file_path, "rb") as file:
                    st.download_button(
                        label="📥 Unduh Snapshot",
                        data=file,
                        file_name=f"snapshot_{last_snapshot['timestamp'].replace(':', '-')}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
    
    with tab3:
        st.header("📊 Analisis Historis Deteksi")
        
        if not st.session_state.detection_history:
            st.info("Belum ada riwayat deteksi. Silakan gunakan fitur deteksi gambar atau real-time terlebih dahulu.")
        else:
            # Filter untuk deteksi video
            video_detections = [d for d in st.session_state.detection_history if d.get('type') == 'video']
            
            # Statistik umum
            st.subheader("📈 Statistik Umum")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Deteksi", len(st.session_state.detection_history))
            
            with col2:
                st.metric("Deteksi Gambar", len([d for d in st.session_state.detection_history if d['type'] == 'image']))
            
            with col3:
                st.metric("Deteksi Video", len(video_detections))
            
            # Statistik khusus video
            if video_detections:
                st.subheader("📊 Statistik Deteksi Video")
                
                # Rata-rata interval
                intervals = [d.get('interval', 10) for d in video_detections]
                avg_interval = sum(intervals) / len(intervals)
                
                # Distribusi emosi
                emotions = [d['emotion'] for d in video_detections]
                emotion_counts = pd.Series(emotions).value_counts().reset_index()
                emotion_counts.columns = ['Emosi', 'Jumlah']
                
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Rata-rata Interval", f"{avg_interval:.1f} detik")
                    
                with cols[1]:
                    # Emosi paling umum
                    most_common_emotion = emotion_counts.iloc[0]['Emosi'] if not emotion_counts.empty else "N/A"
                    st.metric("Emosi Paling Umum", most_common_emotion)
                
                # Grafik distribusi emosi untuk video
                fig = px.pie(
                    emotion_counts, 
                    names='Emosi', 
                    values='Jumlah',
                    hole=0.3,
                    title='Distribusi Emosi (Deteksi Video)',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Riwayat deteksi video
                st.subheader("🕒 Riwayat Deteksi Video")
                st.info(f"Menampilkan {len(video_detections)} deteksi video yang tersimpan")
                
                for item in video_detections[:10]:
                    with st.container():
                        st.markdown(f"<div class='history-item'>", unsafe_allow_html=True)
                        
                        # Header
                        st.markdown(f"""
                        <div class='history-header'>
                            <div>
                                <span class='history-emotion'>{item['emotion']}</span>
                                <span>({item['probability']:.2f})</span>
                            </div>
                            <div>
                                <small>{item['time']} | Interval: {item.get('interval', 10)} detik</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Konten
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.image(item['image'], width=150)
                        
                        with cols[1]:
                            # Tampilkan distribusi probabilitas
                            prob_df = pd.DataFrame({
                                'Emosi': st.session_state.emotions,
                                'Probabilitas': item['all_probabilities']
                            })
                            fig = px.bar(
                                prob_df, 
                                x='Emosi', 
                                y='Probabilitas',
                                color='Emosi',
                                color_discrete_sequence=px.colors.qualitative.Pastel,
                                height=250
                            )
                            fig.update_layout(
                                showlegend=False,
                                margin=dict(l=0, r=0, t=30, b=0),
                                yaxis=dict(range=[0, 1])
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Tombol hapus
                        if st.button("Hapus Deteksi", key=f"del_{item['time']}", use_container_width=True):
                            st.session_state.detection_history = [
                                d for d in st.session_state.detection_history 
                                if not (d.get('time') == item['time'] and d.get('type') == 'video')
                            ]
                            st.experimental_rerun()
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.divider()
            else:
                st.info("Belum ada riwayat deteksi video. Hasil deteksi video disimpan setiap interval waktu tertentu selama deteksi real-time berjalan.")
            
            # Opsi ekspor data
            if st.button("📤 Ekspor Data Historis ke CSV", use_container_width=True):
                history_df = pd.DataFrame(st.session_state.detection_history)
                
                # Hapus kolom gambar untuk CSV
                if 'image' in history_df.columns:
                    history_df = history_df.drop(columns=['image'])
                
                csv = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Unduh CSV",
                    data=csv,
                    file_name="riwayat_deteksi_emosi.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    st.markdown("""
    <div class="footer">
        <p>Sistem Deteksi Emosi Real-time © 2025 | Dibangun dengan Streamlit dan OpenCV</p>
        <p>Dibuat oleh Kelompok 11:</p>
        <ul>
            <li>Lyon Ambrosio Djuanda / 2304130098</li>
            <li>Rafa Afra Zahirah / 2304130099</li>
            <li>Naufal Tipasha Denyana / 2304130115</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()