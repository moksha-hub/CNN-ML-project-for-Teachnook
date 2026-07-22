## 1. Project Description

This project develops a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The dataset contains 60,000 color images belonging to 10 object categories. The model learns image features through convolution and pooling layers and predicts the correct class using a softmax output layer.

## 2. Methodology

The CIFAR-10 dataset is loaded and preprocessed by normalizing pixel values and converting labels into one-hot encoded vectors. A CNN model is then built with convolutional, pooling, flatten, and dense layers. The model is trained using the Adam optimizer and categorical cross-entropy loss, and its performance is evaluated on the test dataset.

## 3. Results and Features

The trained model classifies images into one of the ten CIFAR-10 classes and reports its performance using test accuracy and a classification report. The project also visualizes a sample image, displays the model architecture, and generates predictions for unseen test images, demonstrating the effectiveness of CNNs for image classification.
