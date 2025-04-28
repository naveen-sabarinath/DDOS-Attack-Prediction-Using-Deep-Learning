# DDoS Attack Prediction Using Deep Learning

This repository contains the implementation of a deep learning-based system for detecting Distributed Denial of Service (DDoS) attacks. The approach leverages a **1D Convolutional Neural Network (CNN)** to classify network traffic into three categories: **DDoS-PSH-ACK**, **DDoS-ACK**, and **Benign**. The final model is deployed via a **web application built using Flask**.

## Abstract

Distributed Denial of Service (DDoS) attacks disrupt online services by overwhelming systems with malicious traffic. This project proposes a deep learning solution using 1D CNNs to predict and classify DDoS attacks based on network traffic data. The system is capable of real-time predictions through a Flask-based web interface.

## Features

- Classification of network traffic into:
  - DDoS-PSH-ACK
  - DDoS-ACK
  - Benign
- Deep Learning with 1D Convolutional Neural Networks (CNN)
- Real-time prediction using a Flask Web App
- Data preprocessing, model training, evaluation, and deployment included

## Technologies Used

- Python 3.9
- TensorFlow / Keras
- Flask
- scikit-learn
- pandas
- pickle

## Dataset

- **APA-DDoS-Dataset.csv**: Custom dataset including normal and attack network traffic.
- Key features: `frame.time`, `ip.src`, `ip.dst`, `Label`, etc.

## Model Architecture

- Input: One-dimensional network traffic data
- Layers:
  - Conv1D → ReLU → MaxPooling → Dropout
  - Flatten → Dense Layers → Softmax
- Optimizer: Adam
- Activation: ReLU, Softmax
- Output Classes: 3 (DDoS-PSH-ACK, DDoS-ACK, Benign)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ddos-attack-prediction.git
   cd ddos-attack-prediction
