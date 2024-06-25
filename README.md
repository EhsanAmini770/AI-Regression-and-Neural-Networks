# AI-Regression-and-Neural-Networks

This repository contains various machine learning models applied to multiple datasets. The models implemented include linear regression, polynomial regression, decision tree regression, random forest regression, support vector regression, K-Nearest Neighbors, logistic regression, and neural networks. Below is an overview of the files and their contents.

## Files and Datasets

### Data Files

- **arac_verileri.csv**: Contains car data used for training and testing various regression models.
- **data_diagnosis.csv**: Dataset for diagnosis classification, used for training and testing classification models.
- **data_kalite.csv**: Dataset related to quality measurements, used for regression and classification tasks.
- **data_satınalma.csv**: Dataset for logistic regression tasks.
- **linear_regression_dataset.csv**: Used for linear regression to predict salaries based on experience.
- **satislar.csv**: Used for linear regression to predict sales based on months.
- **veri.csv**: Contains additional data for regression models.
- **veri1.csv**: Used for polynomial regression and other regression models.

### Code Files

- **main.py**: Contains implementations of various regression models (linear, polynomial, decision tree, random forest) and their evaluations.
- **ml_analysis_pipeline.py**: Contains a pipeline for preprocessing datasets and training various models including SVC, KNN, Random Forest, Decision Tree, SVR, and Logistic Regression.
- **mnist_model.h5**: Saved trained model for MNIST dataset using neural networks.

## Machine Learning Models

### Regression Models

#### Linear Regression

- **Salary Prediction**: Predicting salary based on experience from `linear_regression_dataset.csv`.
- **Sales Prediction**: Predicting sales based on months from `satislar.csv`.
- **Car Data Prediction**: Multiple linear regression using `arac_verileri.csv`.

#### Polynomial Regression

- Predicting values using polynomial features of degrees 2 and 4 on `veri1.csv`.

#### Decision Tree Regression

- Implemented using `veri1.csv` and `arac_verileri.csv`.

#### Random Forest Regression

- Implemented using `veri1.csv` and `arac_verileri.csv`.

#### K-Nearest Neighbors Regression

- Implemented using `veri1.csv` and `data_kalite.csv`.

#### Support Vector Regression

- Implemented using `veri1.csv` and `data_kalite.csv`.

### Classification Models

#### Logistic Regression

- Implemented on `data_satınalma.csv`.

#### Support Vector Classifier (SVC)

- Cross-validation and training on `data_diagnosis.csv`.

#### K-Nearest Neighbors (KNN)

- Cross-validation, grid search, and randomized search on `data_diagnosis.csv`.
- KNN Regressor implemented on `data_kalite.csv`.

#### Random Forest Classifier

- Implemented on `data_diagnosis.csv` and `data_kalite.csv`.

#### Decision Tree Classifier

- Implemented on `data_diagnosis.csv` and `data_kalite.csv`.

### Neural Networks

#### MNIST Handwritten Digits

- Fully connected neural network and Convolutional Neural Network (CNN) trained on the MNIST dataset.
- Saved trained model `mnist_model.h5`.
- Sample image prediction using the trained model.

#### CIFAR-10

- Data loading and preprocessing of CIFAR-10 dataset.

## Usage

1. **Setup Environment**:
   - Ensure you have Python installed along with necessary libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow`, `keras`, `cv2`.

2. **Run Scripts**:
   - Execute `main.py` for running regression models.
   - Execute `ml_analysis_pipeline.py` for preprocessing and training various machine learning models.
   - For neural networks, ensure `mnist_model.h5` is in the working directory for loading the trained model.

## Visualizations

The repository contains various plots and visualizations for model predictions and performance evaluations. These plots include scatter plots, line plots, decision trees, and more.

## Contact

For any questions or suggestions, feel free to reach out to [Ehsan Amini](https://github.com/Ehsanamini770).

---

This repository demonstrates the application of various machine learning models on diverse datasets, providing a comprehensive understanding of regression and classification techniques.
