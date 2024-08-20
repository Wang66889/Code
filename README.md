# Aflatoxin Spectral Curve Classification Using Multiverse Optimization-Based Neural Network

This repository contains MATLAB code for classifying aflatoxin spectral curves using a neural network optimized with the Multiverse Optimization (MVO) algorithm. The code combines a one-dimensional convolutional neural network (CNN) with a bidirectional long short-term memory (BiLSTM) network to accurately classify spectral data.

## Project Overview

This project is focused on the classification of aflatoxin spectral curves. The implemented neural network is designed to distinguish between different classes of spectral data by optimizing hyperparameters such as learning rate, number of convolution filters, and number of hidden units using the Multiverse Optimization (MVO) algorithm.

## Prerequisites

- MATLAB R2021a or later
- Deep Learning Toolbox
- A compatible GPU for training (optional but recommended)

## Dataset Preparation

The code requires two datasets in Excel format:

- `train.xlsx`: Contains the training data with spectral input features and corresponding aflatoxin classification labels.
- `val.xlsx`: Contains the validation data with spectral input features and corresponding aflatoxin classification labels.

Ensure these files are located in the same directory as the MATLAB script.

## Code Explanation

1. **Data Loading and Preprocessing**:
   - The training and validation datasets are loaded from Excel files containing spectral curves.
   - The data is shuffled and normalized to the range of 0-1 to improve network performance.
2. **Network Parameters**:
   - Parameters such as the number of input features, dropout rate, and convolution kernel size are defined.
   - The network is designed to classify spectral curves into specific aflatoxin-related categories.
3. **Multiverse Optimization (MVO)**:
   - The MVO algorithm is employed to optimize crucial hyperparameters, such as the learning rate, number of convolution filters, and number of hidden units in the BiLSTM layer.
   - This optimization aims to enhance the classification accuracy of the spectral data.
4. **Network Construction and Training**:
   - The optimized hyperparameters are used to construct and train the neural network.
   - The network architecture includes a 1D convolutional layer, BiLSTM layer, and a fully connected output layer tailored for spectral data classification.
5. **Evaluation**:
   - The trained model is evaluated on both training and validation datasets.
   - Classification accuracy and confusion matrices are generated to assess the performance of the model.

## Running the Code

1. Place the `train.xlsx` and `val.xlsx` files in the same directory as the script.

2. Open MATLAB and run the script.

3. The script will iterate through the MVO optimization process, train the network, and display the results.

   ## Results

   - **Training Accuracy**: Displayed at the end of the script execution, indicating the model's performance on the training data.
   - **Validation Accuracy**: Displayed at the end of the script execution, showing the model's performance on the validation data.
   - **Confusion Matrices**: Confusion matrices for both training and validation datasets are generated to visualize the classification performance.