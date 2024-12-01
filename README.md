# Glioma Grading Prediction

## Overview 

This project aims to predict the grade of gliomas (lower-grade glioma [LGG] and glioblastoma multiforme [GBM]) using clinical and molecular mutation features. The goal is to use feature selection techniques to identify the most important features and improve the accuracy of classification models, reducing the need for costly molecular testing.


## Dataset

The dataset contains clinical and molecular features of gliomas and can be accessed via the following link:  
**[https://drive.google.com/drive/folders/15yd_i1Ur_3MQFTujlTVna9OrCKLQC20m?usp=sharing](#)**  
Please copy the above link and paste in your browser then it will redirect to drive link where the dataset is uploaded

---


## Project Structure

Install my-project with npm

```bash
  .
├── data/                    # Folder containing the dataset
│   └── glioma_data.csv      # Raw dataset
├── notebooks/                # Jupyter notebooks for exploratory analysis
│   └── feature_selection.ipynb  # Code for WoE, RFE, and PCA analysis
├── src/                      # Python scripts
│   ├── preprocess.py         # Data loading and preprocessing
│   ├── feature_selection.py  # Feature selection (WoE, RFE, PCA)
│   ├── models.py             # Model training and evaluation
│   └── ensemble.py           # Stacking classifier implementation
├── requirements.txt          # Python dependencies
└── README.md                 # Project description and instructions

```
    
## Prerequisites

Before running the code, ensure that you have the following installed:

- Python 3.x

- Required Python libraries (listed in `requirements.txt`)


## Install Dependencies

To install the required Python libraries, run:

```bash
  pip install -r requirements.txt

```
 The necessary libraries include:

 - pandas
 - numpy
 - scikit-learn
 - matplotlib
 - seaborn# Setup Instructions

## 1. Data Loading & Preprocessing

The preprocessing code is contained in the preprocess.py script. It handles the loading of the dataset, handling missing values, and selecting relevant features for further analysis.

### To run the preprocessing script:

```bash
  python src/preprocess.py
```
## 2. Feature Selection

Feature selection is performed using three different methods: Weight of Evidence (WoE), Recursive Feature Elimination (RFE), and Principal Component Analysis (PCA).


 - WoE Calculation: WoE is calculated for categorical features to determine their relationship with the target variable (glioma grade).

 - RFE: RFE with Logistic Regression is used to select the most important   features.

 - PCA: PCA is applied to reduce the dimensionality of the dataset while retaining 95% of its variance.
 
 ### To run the feature selection script:

```bash
  python src/feature_selection.py
```
This will generate feature rankings and help visualize which features are most important.

## 3. Model Training & Evaluation

Various classification models are used for predicting glioma grades, including:
 - Random Forest Classifier

 -Support Vector Classifier (SVC)
 - K-Nearest Neighbors (KNN)
 - AdaBoost Classifier
 - Logistic Regression
 
 The script `models.py` handles the training of these models and evaluates them using cross-validation and classification metrics (accuracy, precision, recall, F1-score).


 ### To train and evaluate the models:

```bash
    python src/models.py

```

## 4. Ensemble Learning
An ensemble method, StackingClassifier, is applied to combine the predictions of multiple models (SVM, Random Forest, and AdaBoost) and improve overall prediction accuracy. Logistic Regression is used as the meta-model to combine the base models' outputs.

### To run the ensemble learning script:

```bash
 python src/ensemble.py

```
## 5. Final Prediction
Once the best model has been selected, it will be applied to the test dataset, and the predicted class labels (LGG or GBM) will be saved to a text file.

### Running the Full Pipeline

To run the entire pipeline from data loading to final prediction, you can execute all the scripts sequentially or create a master script that calls each step. For simplicity, you can use the following approach:

```bash
 python src/preprocess.py
python src/feature_selection.py
python src/models.py
python src/ensemble.py
python src/final_prediction.py



```
This will load the data, perform feature selection, train models, evaluate them, and finally produce predictions.
# Glioma Grading Prediction
