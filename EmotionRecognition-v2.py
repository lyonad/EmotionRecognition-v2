import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class PelicanOptimizationAlgorithm:
    def __init__(self, num_pelicans=10, max_iterations=10, lower_bound=1, upper_bound=30):
        self.num_pelicans = num_pelicans
        self.max_iterations = max_iterations
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def initialize_population(self, dim):
        population = []
        for _ in range(self.num_pelicans):
            pelican = {}
            pelican['k'] = np.random.randint(self.lower_bound, self.upper_bound)
            pelican['weights'] = np.random.choice(['uniform', 'distance'])
            lower_pca = max(2, int(0.01 * dim))
            upper_pca = max(lower_pca + 1, min(int(0.5 * dim), 100))
            print(f"Rentang PCA: {lower_pca} - {upper_pca} (dari dimensi {dim})")
            pelican['n_components'] = np.random.randint(lower_pca, upper_pca)
            pelican['fitness'] = float('-inf')
            population.append(pelican)
        return population

    def evaluate_fitness(self, pelican, X_train, y_train, X_val, y_val):
        k = pelican['k']
        weights = pelican['weights']
        n_components = pelican['n_components']
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
        knn.fit(X_train_pca, y_train)
        y_pred = knn.predict(X_val_pca)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy

    def update_position(self, pelican, best_pelican, dim, alpha=0.5):
        if np.random.rand() < alpha:
            delta_k = int(np.random.randint(-2, 3))
            new_k = best_pelican['k'] + delta_k
        else:
            new_k = np.random.randint(self.lower_bound, self.upper_bound)
        pelican['k'] = max(self.lower_bound, min(new_k, self.upper_bound))

        if np.random.rand() < alpha:
            pelican['weights'] = best_pelican['weights']
        else:
            pelican['weights'] = np.random.choice(['uniform', 'distance'])

        lower_pca = max(2, int(0.01 * dim))
        upper_pca = max(lower_pca + 1, min(int(0.5 * dim), 100))
        if np.random.rand() < alpha:
            base_n = best_pelican['n_components']
            delta_n = int(np.random.randint(-5, 6))
            new_n = base_n + delta_n
        else:
            new_n = np.random.randint(lower_pca, upper_pca)
        pelican['n_components'] = max(lower_pca, min(new_n, upper_pca-1))
        return pelican

    def optimize(self, X_train, y_train, X_val, y_val):
        dim = X_train.shape[1]
        print(f"Dimensi fitur input: {dim}")
        population = self.initialize_population(dim)
        for i in range(self.num_pelicans):
            fitness = self.evaluate_fitness(population[i], X_train, y_train, X_val, y_val)
            population[i]['fitness'] = fitness
        best_pelican = max(population, key=lambda x: x['fitness'])
        for iteration in range(self.max_iterations):
            for i in range(self.num_pelicans):
                population[i] = self.update_position(population[i], best_pelican, dim)
                fitness = self.evaluate_fitness(population[i], X_train, y_train, X_val, y_val)
                population[i]['fitness'] = fitness
                if fitness > best_pelican['fitness']:
                    best_pelican = population[i].copy()
            print(f"Iterasi {iteration+1}/{self.max_iterations}, Best Fitness: {best_pelican['fitness']:.4f}, " +
                  f"Best Parameters: k={best_pelican['k']}, weights={best_pelican['weights']}, " +
                  f"n_components={best_pelican['n_components']}")
        return best_pelican

def load_fer_dataset(train_dir, test_dir, debug=True):
    print(f"Memuat dataset dari: {train_dir} dan {test_dir}")
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    if debug:
        print("\nMemeriksa struktur direktori...")
        if not os.path.exists(train_dir):
            print(f"WARNING: Direktori train tidak ditemukan: {train_dir}")
        else:
            print(f"Direktori train ditemukan: {train_dir}")
            print("Isi direktori train:")
            for item in os.listdir(train_dir):
                path = os.path.join(train_dir, item)
                if os.path.isdir(path):
                    print(f"  - {item}/ (folder) - {len(os.listdir(path))} file")
                else:
                    print(f"  - {item} (file)")
        if not os.path.exists(test_dir):
            print(f"WARNING: Direktori test tidak ditemukan: {test_dir}")
        else:
            print(f"Direktori test ditemukan: {test_dir}")
            print("Isi direktori test:")
            for item in os.listdir(test_dir):
                path = os.path.join(test_dir, item)
                if os.path.isdir(path):
                    print(f"  - {item}/ (folder) - {len(os.listdir(path))} file")
                else:
                    print(f"  - {item} (file)")
    first_image_found = False
    total_train_images = 0
    for emotion_idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(train_dir, emotion)
        if os.path.isdir(emotion_dir):
            files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if debug:
                print(f"Memuat {emotion} dari {emotion_dir}: {len(files)} file gambar ditemukan")
            image_count = 0
            for image_file in files:
                img_path = os.path.join(emotion_dir, image_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        if not first_image_found and debug:
                            print(f"Gambar pertama berhasil dimuat: {img_path}")
                            print(f"  Dimensi: {img.shape}")
                            first_image_found = True
                        img = cv2.resize(img, (48, 48))
                        X_train.append(img.flatten())
                        y_train.append(emotion_idx)
                        image_count += 1
                    else:
                        if debug:
                            print(f"WARNING: Gagal memuat gambar: {img_path}")
                except Exception as e:
                    if debug:
                        print(f"ERROR: {e} pada file {img_path}")
            total_train_images += image_count
            if debug:
                print(f"  - Berhasil memuat {image_count} gambar untuk emosi '{emotion}'")
    total_test_images = 0
    for emotion_idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(test_dir, emotion)
        if os.path.isdir(emotion_dir):
            files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if debug:
                print(f"Memuat test {emotion} dari {emotion_dir}: {len(files)} file gambar ditemukan")
            image_count = 0
            for image_file in files:
                img_path = os.path.join(emotion_dir, image_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (48, 48))
                        X_test.append(img.flatten())
                        y_test.append(emotion_idx)
                        image_count += 1
                    else:
                        if debug:
                            print(f"WARNING: Gagal memuat gambar: {img_path}")
                except Exception as e:
                    if debug:
                        print(f"ERROR: {e} pada file {img_path}")
            total_test_images += image_count
            if debug:
                print(f"  - Berhasil memuat {image_count} gambar untuk emosi '{emotion}'")
    if len(X_train) == 0:
        raise ValueError("ERROR: Tidak ada gambar training yang berhasil dimuat! Periksa path direktori dan format gambar.")
    if len(X_test) == 0:
        raise ValueError("ERROR: Tidak ada gambar testing yang berhasil dimuat! Periksa path direktori dan format gambar.")
    print(f"\nTotal gambar training: {total_train_images}")
    print(f"Total gambar testing: {total_test_images}")
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), emotions

def train_model(X_train, y_train, X_test, y_test, emotions):
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("Memulai optimasi POA...")
    start_time = time.time()
    poa = PelicanOptimizationAlgorithm(num_pelicans=10, max_iterations=10, lower_bound=1, upper_bound=30) 
    best_params = poa.optimize(X_train_scaled, y_train_split, X_val_scaled, y_val)
    end_time = time.time()
    print(f"Optimasi selesai dalam {end_time - start_time:.2f} detik")
    print(f"Parameter optimal: k={best_params['k']}, weights={best_params['weights']}, n_components={best_params['n_components']}")
    pca = PCA(n_components=best_params['n_components'])
    X_train_full_scaled = scaler.fit_transform(X_train)
    X_train_pca = pca.fit_transform(X_train_full_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    knn = KNeighborsClassifier(n_neighbors=best_params['k'], weights=best_params['weights'])
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi pada test set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=emotions))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    save_model(knn, pca, scaler, emotions)
    return knn, pca, scaler

def save_model(knn, pca, scaler, emotions, filename='emotion_model_v2.pkl'):
    model_data = {
        'knn': knn,
        'pca': pca,
        'scaler': scaler,
        'emotions': emotions
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model berhasil disimpan ke {filename}")

def load_model(filename='emotion_model_v2.pkl'):
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['knn'], model_data['pca'], model_data['scaler'], model_data['emotions']

def preprocess_face(face_img, target_size=(48, 48)):
    face_img = cv2.resize(face_img, target_size)
    face_vector = face_img.flatten()
    return face_vector

def predict_emotion(face_img, knn, pca, scaler):
    face_vector = preprocess_face(face_img)
    face_vector = face_vector.reshape(1, -1)
    face_scaled = scaler.transform(face_vector)
    face_pca = pca.transform(face_scaled)
    emotion_idx = knn.predict(face_pca)[0]
    emotion_probs = knn.predict_proba(face_pca)[0]
    return emotion_idx, emotion_probs

def real_time_emotion_detection(model_filename='emotion_model_v2.pkl'):
    try:
        print(f"Mencoba memuat model dari {model_filename}...")
        knn, pca, scaler, emotions = load_model(model_filename)
        print("Model berhasil dimuat!")
    except Exception as e:
        print(f"Tidak dapat memuat model: {e}")
        print("Model belum ada atau tidak dapat diakses.")
        return
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses webcam")
        return
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        print(f"ERROR: File face cascade tidak ditemukan di {face_cascade_path}")
        print("\nMendownload file haarcascade dari GitHub...")
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml" 
            face_cascade_path = "haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, face_cascade_path)
            print(f"File cascade berhasil didownload ke {face_cascade_path}")
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
        except Exception as download_error:
            print(f"ERROR: Tidak dapat download cascade: {download_error}")
            print("Deteksi wajah tidak akan berfungsi dengan baik.")
            face_cascade = None
    else:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            print("ERROR: Cascade classifier kosong!")
            face_cascade = None
    emotion_colors = {
        0: (0, 0, 255),
        1: (0, 255, 0),
        2: (255, 0, 255),
        3: (0, 255, 255),
        4: (255, 255, 255),
        5: (255, 0, 0),
        6: (255, 255, 0),
    }
    frame_count = 0
    start_time = time.time()
    fps = 0
    print("Real-time emotion detection dimulai. Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat mengambil frame dari webcam")
            break
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if face_cascade is not None:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        else:
            h, w = gray.shape
            faces = np.array([[w//4, h//4, w//2, h//2]])
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            emotion_idx, emotion_probs = predict_emotion(face_gray, knn, pca, scaler)
            emotion = emotions[emotion_idx]
            probability = emotion_probs[emotion_idx]
            color = emotion_colors.get(emotion_idx, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f"{emotion}: {probability:.2f}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
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
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Tekan 'q' untuk keluar", (10, frame.shape[0]-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.namedWindow("Real-time Emotion Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Real-time Emotion Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Real-time Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def check_dataset_structure(dir_path):
    print(f"\n=== Memeriksa struktur dataset di: {dir_path} ===")
    if not os.path.exists(dir_path):
        print(f"ERROR: Direktori tidak ditemukan: {dir_path}")
        return False
    expected_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    content = os.listdir(dir_path)
    print(f"Isi direktori utama: {content}")
    found_emotions = []
    for item in content:
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            found_emotions.append(item)
    missing_emotions = [e for e in expected_emotions if e not in found_emotions]
    unexpected_folders = [f for f in found_emotions if f not in expected_emotions]
    if missing_emotions:
        print(f"WARNING: Folder emosi berikut tidak ditemukan: {missing_emotions}")
    if unexpected_folders:
        print(f"INFO: Ditemukan folder tidak terduga: {unexpected_folders}")
    print("\nJumlah gambar per folder emosi:")
    total_images = 0
    for emotion in found_emotions:
        emotion_dir = os.path.join(dir_path, emotion)
        image_files = [f for f in os.listdir(emotion_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  - {emotion}: {len(image_files)} gambar")
        if image_files:
            sample_img_path = os.path.join(emotion_dir, image_files[0])
            img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                print(f"    Sampel pertama ({image_files[0]}): {img.shape} piksel")
            else:
                print(f"    WARNING: Tidak dapat membaca gambar sampel: {sample_img_path}")
        total_images += len(image_files)
    print(f"\nTotal gambar yang ditemukan: {total_images}")
    if total_images == 0:
        print("\nREKOMENDASI:")
        print("- Pastikan direktori yang diberikan benar")
        print("- Periksa apakah gambar dalam format yang didukung (.jpg, .jpeg, .png, .bmp)")
        print("- Verifikasi struktur folder (harus sesuai dengan nama emosi)")
        return False
    return total_images > 0

def main():
    train_dir = r'C:\Users\LyonA\Downloads\archive\train'
    test_dir = r'C:\Users\LyonA\Downloads\archive\test'
    print("\n=== SISTEM DETEKSI EMOSI REAL-TIME ===")
    print("\nPilih opsi:")
    print("1. Melatih model baru dengan dataset yang ada")
    print("2. Gunakan model yang sudah ada (jika tersedia)")
    print("3. Periksa struktur dataset")
    option = input("\nPilihan Anda (1/2/3): ")
    if option == '1':
        train_valid = check_dataset_structure(train_dir)
        test_valid = check_dataset_structure(test_dir)
        if not (train_valid and test_valid):
            alternative = input("\nStruktur dataset tidak valid. Apakah ingin memverifikasi path dataset? (y/n): ")
            if alternative.lower() == 'y':
                train_dir = input("Masukkan path direktori training (misalnya: C:\\Users\\LyonA\\Downloads\\archive\\train): ")
                test_dir = input("Masukkan path direktori testing (misalnya: C:\\Users\\LyonA\\Downloads\\archive\\test): ")
                train_valid = check_dataset_structure(train_dir)
                test_valid = check_dataset_structure(test_dir)
        if not (train_valid and test_valid):
            print("\nDATASET TIDAK VALID: Tidak dapat melanjutkan pelatihan model.")
            print("Coba opsi 3 untuk memeriksa struktur dataset.")
            return
        try:
            print("\nMemuat dataset...")
            X_train, y_train, X_test, y_test, emotions = load_fer_dataset(train_dir, test_dir)
            if len(X_train) == 0 or len(X_test) == 0:
                print("ERROR: Dataset kosong, tidak dapat melatih model.")
                return
            print(f"Dataset dimuat: {X_train.shape[0]} sampel training, {X_test.shape[0]} sampel testing")
            print("\nMelatih model...")
            knn, pca, scaler = train_model(X_train, y_train, X_test, y_test, emotions)
            print("\nModel berhasil dilatih!")
            run_detection = input("\nJalankan deteksi emosi real-time sekarang? (y/n): ")
            if run_detection.lower() == 'y':
                real_time_emotion_detection()
        except Exception as e:
            print(f"ERROR saat melatih model: {e}")
    elif option == '2':
        model_path = input("\nMasukkan path model (default: emotion_model_v2.pkl): ") or "emotion_model_v2.pkl"
        if not os.path.exists(model_path):
            print(f"File model tidak ditemukan: {model_path}")
            print("Tidak ada model yang dapat digunakan. Program berhenti.")
        else:
            real_time_emotion_detection(model_filename=model_path)
    elif option == '3':
        custom_train = input("\nMasukkan path direktori training (kosongkan untuk default): ")
        custom_test = input("Masukkan path direktori testing (kosongkan untuk default): ")
        train_dir = custom_train if custom_train else train_dir
        test_dir = custom_test if custom_test else test_dir
        check_dataset_structure(train_dir)
        check_dataset_structure(test_dir)
    else:
        print("Pilihan tidak valid")

if __name__ == "__main__":
    main()