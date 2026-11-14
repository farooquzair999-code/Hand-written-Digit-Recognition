# Handwritten Digit Recognition Using CNN

This project implements a **Convolutional Neural Network (CNN)** to recognize and classify handwritten digits (`0–9`). Using deep learning with TensorFlow and Keras, the model learns spatial features from grayscale digit images and accurately predicts the written number.

## 📌 Project Overview
Handwriting varies significantly from person to person in style, slant, and thickness. In this notebook, a deep learning model is trained on the MNIST dataset of handwritten digits. Once trained, the model takes an unseen handwritten digit image, predicts the number, and displays the visual result.

## 📊 Dataset & Preprocessing
* **Dataset:** Built-in **MNIST** handwritten digits dataset (`0–9`) loaded via `tf.keras.datasets.mnist`.
* **Image Dimensions:** `28x28` grayscale images reshaped to `(28, 28, 1)` for convolutional processing.
* **Normalization:** Pixel intensity values (originally `0–255`) are divided by `255.0` to scale them between `0` and `1` for stable and efficient training.

## 🧠 Model Architecture & Results
A Sequential Convolutional Neural Network (CNN) was built and trained for **5 epochs** using the `adam` optimizer and `sparse_categorical_crossentropy` loss function:

1. **Reshape Layer:** Formats input images to `(28, 28, 1)`.
2. **Conv2D Layer:** 32 filters with a `(3, 3)` kernel and `relu` activation to extract image features.
3. **MaxPooling2D Layer:** `(2, 2)` pool size to downsample feature maps.
4. **Flatten Layer:** Converts 2D feature maps into a 1D vector.
5. **Dense Layer:** 64 neurons with `relu` activation.
6. **Output Layer:** 10 neurons with `softmax` activation representing digits `0` through `9`.

* **Final Test Accuracy:** **98.58%** (Test Loss: `0.0464`)

## 🛠️ Tools & Libraries Used
* **Python 3** — Core programming language.
* **TensorFlow / Keras** — Used to load the MNIST dataset, build the Sequential CNN architecture, compile, train, and evaluate the model.
* **NumPy** — Used for array reshaping and extracting the predicted digit class (`np.argmax`).
* **Matplotlib** — Used to visualize test images alongside the model's predicted output.

## 🔍 Sample Prediction (Inference)
The notebook picks a sample handwritten digit image from the test dataset (`x_test[0]`), runs it through the trained CNN, and plots the image using `matplotlib` with the predicted digit displayed as the title.

## 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd your-repo-name
   ```

2. **Install required libraries:**
   ```bash
   pip install tensorflow numpy matplotlib jupyter
   ```

3. **Run the notebook:**
   Open the `.ipynb` file in **Google Colab**, **Jupyter Notebook**, or **VS Code** and run all cells sequentially.
