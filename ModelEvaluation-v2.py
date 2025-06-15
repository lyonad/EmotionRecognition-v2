import numpy as np
import os
import cv2
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Default directories
DEFAULT_TEST_DIR = r'C:\Users\LyonA\Downloads\archive\test'
DEFAULT_MODEL_PATH = 'emotion_model_v2.pkl'
EVALUATION_DIR = 'evaluation_results'

def load_test_dataset(test_dir, emotions, debug=True):
    """
    Load test dataset from directory structure
    
    Parameters:
    test_dir (str): Path to test dataset directory
    emotions (list): List of emotion labels
    debug (bool): Flag to show debug information
    
    Returns:
    tuple: (X_test, y_test) as numpy arrays
    """
    X_test = []
    y_test = []
    total_images = 0
    
    print(f"\nLoading test dataset from: {test_dir}")
    for emotion_idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(test_dir, emotion)
        if os.path.isdir(emotion_dir):
            files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if debug:
                print(f"Loading {emotion} from {emotion_dir}: {len(files)} files found")
            
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
                            print(f"WARNING: Failed to load image: {img_path}")
                except Exception as e:
                    if debug:
                        print(f"ERROR: {e} on file {img_path}")
            
            total_images += image_count
            if debug:
                print(f"  - Successfully loaded {image_count} images for emotion '{emotion}'")
    
    if len(X_test) == 0:
        raise ValueError("ERROR: No test images loaded! Check directory path and image formats.")
    
    print(f"Total test images loaded: {total_images}")
    return np.array(X_test), np.array(y_test)

def evaluate_model(model_filename, test_dir):
    """
    Evaluate model performance and save reports to evaluation_results folder
    
    Parameters:
    model_filename (str): Path to trained model file (.pkl)
    test_dir (str): Path to test dataset directory
    """
    # Create evaluation directory if not exists
    if not os.path.exists(EVALUATION_DIR):
        os.makedirs(EVALUATION_DIR)
    
    # Load trained model
    try:
        print(f"\nLoading model from: {model_filename}")
        with open(model_filename, 'rb') as f:
            model_data = pickle.load(f)
        
        knn = model_data['knn']
        pca = model_data['pca']
        scaler = model_data['scaler']
        emotions = model_data['emotions']
        
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load test dataset
    try:
        X_test, y_test = load_test_dataset(test_dir, emotions)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Preprocess test data
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Make predictions
    y_pred = knn.predict(X_test_pca)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_file = os.path.join(EVALUATION_DIR, 'overall_accuracy.txt')
    with open(accuracy_file, 'w') as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=emotions, digits=4)
    report_file = os.path.join(EVALUATION_DIR, 'classification_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions, 
                yticklabels=emotions,
                annot_kws={"size": 14})
    plt.title('Confusion Matrix - Emotion Detection', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Per-class accuracy analysis
    class_accuracies = []
    class_report = []
    for i, emotion in enumerate(emotions):
        idx = (y_test == i)
        correct = np.sum(y_test[idx] == y_pred[idx])
        total = np.sum(idx)
        acc = correct / total if total > 0 else 0
        class_accuracies.append(acc)
        class_report.append(f"{emotion}: {acc:.4f} ({correct}/{total} correct)")
    
    # Save per-class accuracy report
    class_file = os.path.join(EVALUATION_DIR, 'per_class_accuracy.txt')
    with open(class_file, 'w') as f:
        f.write("\n".join(class_report))
    
    # Visualize per-class accuracy
    plt.figure(figsize=(12, 6))
    bars = plt.bar(emotions, class_accuracies, color='skyblue')
    plt.title('Detection Accuracy per Emotion Class', fontsize=16)
    plt.xlabel('Emotion', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', 
                ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, 'per_class_accuracy.png'), dpi=300)
    plt.close()
    
    print(f"\nEvaluation completed! Results saved in '{EVALUATION_DIR}' folder")

if __name__ == "__main__":
    print("Emotion Detection Model Evaluation")
    print("=" * 40)
    
    # Use default paths
    model_path = input(f"Path to model file [Enter for default '{DEFAULT_MODEL_PATH}']: ").strip() or DEFAULT_MODEL_PATH
    test_data_path = input(f"Path to test dataset [Enter for default '{DEFAULT_TEST_DIR}']: ").strip() or DEFAULT_TEST_DIR
    
    print("\nStarting evaluation with:")
    print(f"- Model: {model_path}")
    print(f"- Test dataset: {test_data_path}")
    print("=" * 40)
    
    evaluate_model(model_path, test_data_path)