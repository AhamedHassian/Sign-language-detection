# ASL Sign Language Detection

This project trains a deep learning model to recognize American Sign Language (ASL) letters from images and deploys it to detect hand signs in real-time using a webcam.

## 📌 Installation

To run this script, install the required dependencies using:

```bash
pip install opencv-python numpy tensorflow keras
```

## 📂 Dataset

The dataset used is `ASL Alphabet Dataset`, which contains images of ASL letters.

- **Dataset Location:** `D:\ML\ASL\asl_alphabet_train`
- **Classes:** 29 (A-Z + space + delete + nothing)

## 🚀 Model Training

The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. It consists of:

- 3 Convolutional Layers with ReLU activation
- MaxPooling layers to reduce dimensionality
- Fully Connected Dense layers
- Output layer with Softmax activation (29 classes)

### 🔹 Steps to Train

1. Load the dataset using `ImageDataGenerator`.
2. Train the model on images of shape `(64, 64, 3)`.
3. Compile using Adam optimizer and categorical cross-entropy loss.
4. Train for **10 epochs** with validation.

Run the training script:

```bash
python main.ipynb
```

After training, the model is saved as:

```bash
model.save('asl_sign_language_model.keras')
```

## 🎥 Real-Time Sign Detection

The model is used for real-time sign language detection using OpenCV.

Run the webcam script:

```bash
python run_camera.ipynb
```

### 🔹 How It Works:

1. Captures video from webcam.
2. Preprocesses the frame (resize, normalize, color conversion).
3. Predicts the ASL letter using the trained model.
4. Displays the detected letter on the screen.
5. Press **'q'** to exit the webcam.

## 📌 Example Output

When the webcam detects a sign, it displays:

```
Predicted: A
```

## 🛠 Dependencies

- TensorFlow
- Keras
- OpenCV
- NumPy

## 📜 License

This project is open-source and available for modification and enhancement.

---

Developed for ASL Sign Language Detection.
