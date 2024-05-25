import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Function to load and preprocess data_diagnosis.csv
def preprocess_data_diagnosis(file_path):
    data = pd.read_csv(file_path)
    data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    y = data.diagnosis.values
    x = data.drop(['diagnosis'], axis=1)
    x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
    
    return x_normalized, y

# Function to perform cross-validation with SVC
def cross_validate_svc(x, y):
    svc = SVC(kernel='rbf')
    scores = cross_val_score(svc, x, y, cv=10)
    print("SVC Cross-validation mean score: {:.2f}%".format(scores.mean() * 100))
    print("SVC Cross-validation std: {:.2f}%".format(scores.std() * 100))

# Function to perform grid search with KNN
def grid_search_knn(x, y):
    knn = KNeighborsClassifier()
    params = {"n_neighbors": np.arange(1, 100)}
    knn_cv = GridSearchCV(knn, params, cv=10)
    knn_cv.fit(x, y)
    print("Best K for KNN:", knn_cv.best_params_)
    print("Best KNN score: {:.2f}%".format(knn_cv.best_score_ * 100))

# Function to perform randomized search with KNN
def randomized_search_knn(x, y):
    knn = KNeighborsClassifier()
    params = {"n_neighbors": np.arange(1, 100)}
    knn_random = RandomizedSearchCV(knn, params, cv=10, n_iter=10, random_state=5)
    knn_random.fit(x, y)
    print("Best K for KNN with RandomizedSearchCV:", knn_random.best_params_)
    print("Best KNN score with RandomizedSearchCV: {:.2f}%".format(knn_random.best_score_ * 100))

# Function to load and preprocess data_kalite.csv
def preprocess_data_kalite(file_path):
    data = pd.read_csv(file_path)
    x = data.iloc[:, 0:12].values
    y = data.iloc[:, 12].values

    le = LabelEncoder()
    x[:, 0] = le.fit_transform(x[:, 0])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    return x_scaled, y

# Function to perform Random Forest on data_kalite.csv
def random_forest_kalite(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=7)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print("Random Forest score on data_kalite.csv: {:.2f}%".format((y_test == y_pred).mean() * 100))

# Function to perform KNN Regressor on data_kalite.csv
def knn_regressor_kalite(x_train, x_test, y_train, y_test):
    knn_reg = KNeighborsRegressor(n_neighbors=1)
    knn_reg.fit(x_train, y_train)
    y_pred = knn_reg.predict(x_test)
    print("KNN Regressor score on data_kalite.csv: {:.2f}%".format((y_test == y_pred).mean() * 100))

# Function to perform SVR on data_kalite.csv
def svr_kalite(x_train, x_test, y_train, y_test):
    svr_reg = SVR(kernel='linear')
    svr_reg.fit(x_train, y_train)
    y_pred = svr_reg.predict(x_test)
    print("SVR score on data_kalite.csv: {:.2f}%".format((y_test == y_pred).mean() * 100))

# Function to perform Decision Tree Regressor on data_kalite.csv
def decision_tree_regressor_kalite(x_train, x_test, y_train, y_test):
    dtr = DecisionTreeRegressor()
    dtr.fit(x_train, y_train)
    y_pred = dtr.predict(x_test)
    print("Decision Tree Regressor score on data_kalite.csv: {:.2f}%".format((y_test == y_pred).mean() * 100))

# Function to perform SVC on data_diagnosis.csv
def svc_poly_diagnosis(x_train, x_test, y_train, y_test):
    svc_poly = SVC(kernel='poly')
    svc_poly.fit(x_train, y_train)
    y_pred = svc_poly.predict(x_test)
    print("SVC score with poly kernel on data_diagnosis.csv: {:.2f}%".format(svc_poly.score(x_test, y_test) * 100))

# Function to perform Random Forest Classifier on data_diagnosis.csv
def random_forest_diagnosis(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=6)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print("Random Forest score on data_diagnosis.csv: {:.2f}%".format(rf.score(x_test, y_test) * 100))

# Function to perform Decision Tree Classifier on data_diagnosis.csv
def decision_tree_diagnosis(x_train, x_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    print("Decision Tree score on data_diagnosis.csv: {:.2f}%".format(dt.score(x_test, y_test) * 100))

# Function to plot data_diagnosis
def plot_data_diagnosis(data):
    M = data[data.diagnosis == 1]
    B = data[data.diagnosis == 0]
    plt.scatter(M.radius_mean, M.texture_mean, color="red", label="Malignant", alpha=0.3)
    plt.scatter(B.radius_mean, B.texture_mean, color="green", label="Benign", alpha=0.3)
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")
    plt.legend()
    plt.show()

# Function to perform KNN Regression on veri1.csv
def knn_regression_veri1(file_path):
    veriler = pd.read_csv(file_path)
    x = veriler.iloc[:, 1:2]
    y = veriler.iloc[:, 2:]

    X = x.values
    Y = y.values

    knn_reg = KNeighborsRegressor(n_neighbors=2)
    knn_reg.fit(X, Y)

    plt.scatter(X, Y, color='red')
    plt.plot(X, knn_reg.predict(X), color='blue')
    plt.show()

    print('Prediction for 6.6:', knn_reg.predict([[6.6]]))

    X_grid = np.arange(X.min(), X.max(), 0.01).reshape(-1, 1)

    plt.scatter(X, Y, color='red')
    plt.plot(X_grid, knn_reg.predict(X_grid), color='blue')
    plt.title('KNN')
    plt.xlabel('Eğitim Seviyesi')
    plt.ylabel('Maaş')
    plt.show()

# Function to perform SVR on veri1.csv
def svr_veri1(file_path):
    veriler = pd.read_csv(file_path)
    x = veriler.iloc[:, 1:2]
    y = veriler.iloc[:, 2:]

    X = x.values
    Y = y.values

    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)

    svr_reg = SVR(kernel='poly')
    svr_reg.fit(X_scaled, Y)

    plt.scatter(X, Y, color='red')
    plt.plot(X, svr_reg.predict(X_scaled), color='blue')
    plt.title('SVR')
    plt.xlabel('Eğitim Seviyesi')
    plt.ylabel('Maaş')
    plt.show()

    print('Prediction for 6.6 with SVR:', svr_reg.predict(scalar.transform([[6.6]])))

# Function to perform Logistic Regression on data_satınalma.csv
def logistic_regression_satinalma(file_path):
    dataset = pd.read_csv(file_path)
    x = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = LogisticRegression()
    classifier.fit(X_train_scaled, y_train)

    print('Logistic Regression Score: {:.2f}%'.format(classifier.score(X_test_scaled, y_test) * 100))

    y_pred = classifier.predict(X_test_scaled)
    print('Mean accuracy: {:.2f}%'.format((y_test == y_pred).mean() * 100))

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', cm)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Data paths
    data_diagnosis_path = "data_diagnosis.csv"
    data_kalite_path = "data_kalite.csv"
    veri1_path = "veri1.csv"
    data_satinalma_path = "data_satınalma.csv"

    # Preprocess and analyze data_diagnosis
    x_diagnosis, y_diagnosis = preprocess_data_diagnosis(data_diagnosis_path)
    cross_validate_svc(x_diagnosis, y_diagnosis)
    grid_search_knn(x_diagnosis, y_diagnosis)
    randomized_search_knn(x_diagnosis, y_diagnosis)
    
    x_train_diag, x_test_diag, y_train_diag, y_test_diag = train_test_split(x_diagnosis, y_diagnosis, test_size=0.3, random_state=42)
    svc_poly_diagnosis(x_train_diag, x_test_diag, y_train_diag, y_test_diag)
    random_forest_diagnosis(x_train_diag, x_test_diag, y_train_diag, y_test_diag)
    decision_tree_diagnosis(x_train_diag, x_test_diag, y_train_diag, y_test_diag)

    # Plot data_diagnosis
    data_diagnosis = pd.read_csv(data_diagnosis_path)
    plot_data_diagnosis(data_diagnosis)

    # Preprocess and analyze data_kalite
    x_kalite, y_kalite = preprocess_data_kalite(data_kalite_path)
    x_train_kalite, x_test_kalite, y_train_kalite, y_test_kalite = train_test_split(x_kalite, y_kalite, test_size=0.2, random_state=42)
    random_forest_kalite(x_train_kalite, x_test_kalite, y_train_kalite, y_test_kalite)
    knn_regressor_kalite(x_train_kalite, x_test_kalite, y_train_kalite, y_test_kalite)
    svr_kalite(x_train_kalite, x_test_kalite, y_train_kalite, y_test_kalite)
    decision_tree_regressor_kalite(x_train_kalite, x_test_kalite, y_train_kalite, y_test_kalite)

    # KNN Regression and SVR on veri1
    knn_regression_veri1(veri1_path)
    svr_veri1(veri1_path)

    # Logistic Regression on data_satınalma
    logistic_regression_satinalma(data_satinalma_path)
