# Face Recognition Comparison AI

A premium Flutter application that performs high-precision face comparison using **TensorFlow Lite (FaceNet)** and **Google ML Kit**.

## ✨ Features
- **Face Detection Integration**: Automatically verifies and crops faces using Google ML Kit before running recognition, eliminating false positives from non-face images (like QR codes).
- **High-Accuracy Recognition**: Uses the FaceNet model with L2 normalization for reliable Euclidean distance benchmarks.
- **Premium UI**: Modern dark-themed design featuring glassmorphism, indigo gradients, and smooth loading animations.
- **Auto-Detection**: Dynamically adjusts to your TFLite model's input/output shapes (supports 128 and 512 length embeddings).
- **Dual-Source Picking**: Select images from both the Gallery and the Camera.

## 🛠️ Tech Stack
- **Framework**: Flutter
- **Machine Learning**: TensorFlow Lite (`tflite_flutter`)
- **Face Detection**: Google ML Kit (`google_mlkit_face_detection`)
- **Image Processing**: `image` package
- **Models**: FaceNet (TFLite)

## 🚀 Getting Started

### Prerequisites
- Flutter SDK (>= 3.6.0)
- Android Studio / VS Code
- A physical device (recommended for camera and TFLite performance)

### Setup
1. Clone the repository:
   ```bash
   git clone [your-repo-url]
   ```
2. Install dependencies:
   ```bash
   flutter pub get
   ```
3. Place your `facenet.tflite` model in the `assets/` folder.
4. Run the app:
   ```bash
   flutter run
   ```

## 📊 How it Works
1. **Detection**: ML Kit scans the image for a face.
2. **Cropping**: If found, the image is cropped exactly to the face region.
3. **Inference**: The cropped face is normalized and passed through FaceNet.
4. **Comparison**: The Euclidean distance between two face embeddings is calculated. (Threshold: < 1.0 for a match).

---
*Developed for high-fidelity face recognition on mobile.*
