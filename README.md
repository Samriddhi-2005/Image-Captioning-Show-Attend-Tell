# 👁️🗣️ Show, Attend, and Tell: Image Captioning with Visual Attention

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Red)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Computer_Vision-brightgreen)

## 📌 Project Overview
This project implements the famous **"Show, Attend and Tell"** neural network architecture to automatically generate descriptive captions for images. 

Unlike standard image captioning models that simply guess the next word, this model uses an **Attention Mechanism**. As it generates each word of the caption, it calculates mathematical weights to determine exactly which pixels of the image it needs to "look at."

### 🌟 Example Output & Attention Heatmap
![Attention Heatmap Example](heatmap.jpeg)

## 🧠 Architecture
The model is split into two main components:
1. **The Encoder ("Show"):** A Convolutional Neural Network (CNN) that processes the input image and extracts a rich set of visual features.
2. **The Decoder ("Attend and Tell"):** A Long Short-Term Memory (LSTM) network equipped with an Attention Mechanism. It generates the sentence word-by-word while visually focusing on different parts of the image feature map.

## 🗄️ Dataset
This model was trained from scratch using the **Flickr8k Dataset**, which contains 8,000 images, each paired with 5 different descriptive captions.
