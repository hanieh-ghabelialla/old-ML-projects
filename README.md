# classifying-fashion-mnist-dataset

The code is implemented for classifying the fashion Mnist dataset. the deep learning architecture used in this project is CNN. I used google Colab environment for implementing this code. below is the link of reference I utilized to do the project.
.
.
.
.
.
.
.
.
.
.
.
.
.
https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/

## The SVM Project (Breast Cancer Classification)

In this project we want to classify a two class dataset called Breast Cancer Coimbra dataset which one class is related to patients with breast cancers and the other class is related to healthy controls. You can see the more details about features and and samples in the link below. https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra. the code is implemented in google colab and you can see more explanation about the code as you open it. the csv format of data is also attached.

## ⚠️ Important Notice About Preprocessing

**The preprocessing functions in this repository are dataset-specific and NOT interchangeable!**

### Breast Cancer Preprocessing
The `preprocessing_the_data()` function in `breast_cancer.ipynb` is **ONLY** designed for the Breast Cancer Coimbra dataset. It makes specific assumptions:
- Exactly 10 columns (9 features + 1 class label)
- Class label in column 9 with values 1 and 2
- All numerical features, no missing values
- **This will NOT work for IBM transaction datasets or other data structures**

### IBM Transaction Dataset Support
For IBM transaction datasets, use the specialized preprocessing functions in:
- `ibm_transaction_preprocessing.py` - Complete preprocessing pipeline for transaction data
- `ibm_transaction_example.ipynb` - Example usage and comparison

### Key Differences
IBM transaction datasets require different preprocessing because they have:
- Mixed data types (numerical, categorical, temporal)
- Class imbalance (fraud is rare)
- Missing values
- High cardinality categorical features
- Time-series dependencies

For detailed analysis and recommendations, see `PREPROCESSING_ANALYSIS.md`.
