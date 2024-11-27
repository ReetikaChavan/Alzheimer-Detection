# Alzheimer's Disease Prediction

## Overview

This project aims to predict Alzheimer's disease stages using medical imaging data. The model is based on **ResNet50**, a powerful deep learning architecture, fine-tuned to classify Alzheimer's stages into categories such as Mild Demented, Moderate Demented, and Non-Demented. This system can assist in early diagnosis and monitoring of Alzheimer's progression.

## Motivation

Alzheimer's disease is a leading cause of dementia, affecting millions worldwide. Early detection and stage classification can help healthcare professionals provide timely interventions. This project applies machine learning and deep learning to medical image classification, enabling accurate predictions for Alzheimer's disease stages.

## Methodology

The Alzheimer’s Disease Prediction system follows these steps:

- **Data Collection**: The dataset contains brain scan images (MRI or CT scans) categorized into four stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented.
- **Pre-processing**: Images are pre-processed by resizing, normalization, and augmentation techniques to enhance model accuracy.
- **ResNet50 Model**: We leverage the pre-trained **ResNet50** architecture, fine-tuned on the Alzheimer's dataset, to classify images into different stages of Alzheimer’s disease.
- **Model Training**: The model is trained using the pre-processed dataset, and performance metrics like accuracy and loss are tracked.
- **Prediction**: The model predicts the stage of Alzheimer's disease based on input medical images.

## Repository Structure

- **dataset/**: Contains the dataset of medical images used for training and testing.
- **model.py**: Script that loads the dataset, pre-processes images, and trains the ResNet50 model. The script will generate an `.h5` model file.
- **app.py**: Streamlit application that allows users to upload images and get predictions for Alzheimer's disease stages.
- **model/**: Folder containing the saved trained model (`.h5`) file.
- **requirements.txt**: List of required dependencies for running the project.

## Implementation Steps

### 1. Navigate to the project directory

```bash
cd Alzheimer-Disease-Prediction

### 2. Run the model.py file
This will load the dataset, pre-process images, and train the ResNet50 model. After training, the model will be saved as an `.h5` file inside the `model/` directory.

```bash
python model.py

### 3. Run the Streamlit application
Launch the app to interact with the model and make predictions on new medical images.

```bash
streamlit run app.py

## Acknowledgments
- The model uses the ResNet50 architecture, pre-trained on ImageNet and fine-tuned on the Alzheimer's dataset.
- The dataset used in this project is publicly available for academic research purposes.





