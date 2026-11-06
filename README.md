# 🩺 Medical Symptom Classifier using Artificial Neural Networks (ANN)

An educational Machine Learning and Deep Learning project that uses an Artificial Neural Network (ANN) to classify disease labels based on selected symptoms.

The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, ANN model development, model evaluation, and an interactive web application built using Streamlit.

> **Disclaimer:** This project is developed strictly for educational and learning purposes. It is not a medical diagnostic system and must not be used to diagnose diseases, make treatment decisions, or replace professional medical advice.

---

## 📌 Project Overview

The Medical Symptom Classifier demonstrates how Artificial Neural Networks can be applied to a multi-class classification problem.

Users can select one or more symptoms through the Streamlit interface. The trained ANN model processes the selected symptoms and displays its top predicted disease classes with their corresponding model scores.

### Project Objectives

- Understand and analyse a structured medical symptom dataset.
- Perform data cleaning and exploratory data analysis.
- Convert categorical symptoms into numerical features.
- Prepare data for ANN model training.
- Build and train an Artificial Neural Network using TensorFlow and Keras.
- Evaluate model performance using classification metrics.
- Develop an interactive application using Streamlit.
- Understand the complete end-to-end machine learning workflow.

---

## 🗂️ Project Structure

```text
medical-symptom-classifier/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── images/
│   ├── average_symptoms_per_disease.png
│   ├── disease_distribution.png
│   ├── symptoms_per_record.png
│   └── top_20_symptoms.png
│
├── models/
│   ├── disease_ann_model.keras
│   ├── feature_scaler.pkl
│   ├── label_encoder.pkl
│   ├── symptom_binarizer.pkl
│   └── symptom_columns.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_Data_Preparation.ipynb
│   ├── 05_ANN_Model_Building.ipynb
│   └── 06_ANN_Model_Evaluation.ipynb
│
├── reports/
│   ├── dataset_overview.txt
│   ├── eda_summary.txt
│   └── ann_evaluation_report.txt
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📊 Dataset Information

**Dataset:** Disease Symptom Description Dataset

**Source:** Kaggle

Dataset link:  
https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

The dataset contains disease labels and associated symptom columns.

### Dataset characteristics

- Target variable: `Disease`
- Input features: Symptom columns
- Number of symptom columns in the original dataset: 17
- Original dataset shape: 4,920 records × 18 columns
- Commonly reported disease classes: 41

The dataset also provides supporting information files, including symptom severity, disease descriptions, and precautions.

**Note:** Download the dataset from Kaggle and place the required CSV files in `data/raw/` before running the notebooks. The raw dataset is not included in this repository.

---

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Data preprocessing and evaluation |
| TensorFlow | Deep learning framework |
| Keras | ANN model development |
| Joblib | Saving preprocessing objects |
| Streamlit | Interactive web application |
| Jupyter Notebook | Experimentation and development |
| Git & GitHub | Version control |

---

## 🔍 Project Workflow

### 1. Data Understanding

- Load the original dataset.
- Inspect rows, columns, and data types.
- Identify the target variable and symptom features.
- Examine missing values and duplicate records.
- Understand the distribution of disease classes.

### 2. Data Cleaning and Exploratory Data Analysis

- Standardize column names and text values.
- Handle missing symptom values.
- Examine duplicate records and repeated symptom patterns.
- Analyse disease distribution.
- Identify frequently occurring symptoms.
- Visualize the number of symptoms per record.

### 3. Feature Engineering

- Extract and organize symptom information.
- Convert symptom selections into numerical features.
- Prepare the feature representation required by the ANN model.

### 4. Data Preparation

- Prepare input features and target labels.
- Encode disease labels.
- Split the data into training, validation, and testing sets.
- Apply the required preprocessing transformations.

### 5. ANN Model Building

- Build an Artificial Neural Network using TensorFlow and Keras.
- Train the model using the prepared training data.
- Monitor training and validation performance.
- Save the trained model and preprocessing objects.

### 6. Model Evaluation

- Evaluate the trained model on test data.
- Analyse classification performance.
- Review relevant evaluation metrics and visualizations.

### 7. Streamlit Deployment

- Load the trained ANN model.
- Load the saved