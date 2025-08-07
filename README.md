# PROJECT_2025
The MNIST dataset is a popular dataset used for training and testing in the field of machine learning for handwritten digit recognition. The article aims to explore the MNIST dataset, its characteristics and its significance in machine learning.
The MNIST dataset is a foundational resource in deep learning, widely used for training and testing models in handwritten digit recognition.
Overview of MNIST
The MNIST dataset (Modified National Institute of Standards and Technology) consists of 70,000 grayscale images of handwritten digits (0-9), with 60,000 images for training and 10,000 for testing. Each image is 28x28 pixels in size, making it a standard benchmark for evaluating machine learning algorithms, particularly in image processing and computer vision.
Origin of the MNIST Dataset
The MNIST dataset, which currently represents a primary input for many tasks in image processing and machine learning, can be traced back to the National Institute of Standards and Technology (NIST). NIST, a US government agency focused on measurement science and standards, curates various datasets, including two particularly relevant to handwritten digits:

#  Handwritten Digit Recognition using Deep Learning (MNIST Dataset)

This project is a simple deep learning model built using TensorFlow and Keras to recognize handwritten digits (0-9) from the popular MNIST dataset. The model is trained using a fully connected neural network and can be extended to use Convolutional Neural Networks (CNNs) for better accuracy.

---

## Dataset

- **Name:** MNIST (Modified National Institute of Standards and Technology)
- **Description:** 28x28 grayscale images of handwritten digits.
- **Training samples:** 60,000
- **Testing samples:** 10,000
- **Classes:** 10 (Digits from 0 to 9)

---

##  Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Jupyter Notebook / Google Colab

---

##  How It Works

1. **Load Data**  
   Load MNIST dataset from `tensorflow.keras.datasets`.

2. **Preprocess Data**  
   - Normalize image pixel values to range [0, 1]  
   - Reshape images to fit the model input  
   - Convert labels to one-hot encoded vectors using `to_categorical()`

3. **Build Model**  
   - Flatten input layer (28x28 → 784)
   - Add Dense layer with ReLU activation
   - Output layer with Softmax activation for 10 classes

4. **Compile Model**  
   - Loss: `categorical_crossentropy`  
   - Optimizer: `adam`  
   - Metric: `accuracy`

5. **Train Model**  
   - Use `model.fit()` on training data

6. **Evaluate Model**  
   - Use `model.evaluate()` on test data

---

## 📈 Model Performance

| Metric | Value |

| Accuracy | ~97% (on test data) |
| Loss | Low (depends on training settings) |

---

##  Sample Output

> Model predicts:  
> Actual Label:   
>  Correct prediction

---
<img width="1920" height="1080" alt="MNIST_NEURAL_IMAAGES" src="https://github.com/user-attachments/assets/aa459fde-f3da-4342-8913-8e4a0e455516" />

<img width="800" height="600" alt="MNIST IMAGES" src="https://github.com/user-attachments/assets/d1edf288-45ce-478d-ae31-871a86a1df18" />


                                                              
