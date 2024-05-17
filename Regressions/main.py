import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Multiple Linear Regression
df = pd.read_csv("data.csv", sep=";")
x = df.iloc[:, [0, 2]].values  # Adjust the indices based on your columns
y = df['maas'].values.reshape(-1, 1)

mlr = LinearRegression()
mlr.fit(x, y)

print("Multiple Linear Regression")
print("b0: ", mlr.intercept_)
print("b1, b2: ", mlr.coef_)
print("Prediction for [10, 35]: ", mlr.predict(np.array([[10, 35]])))

# Polynomial Regression
veriler = pd.read_csv("veri1.csv")
x = veriler.iloc[:, 1:2].values
y = veriler.iloc[:, 2:].values

lin_reg = LinearRegression()
lin_reg.fit(x, y)

poly_reg2 = PolynomialFeatures(degree=2)
x_poly2 = poly_reg2.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2, y)

poly_reg4 = PolynomialFeatures(degree=4)
x_poly4 = poly_reg4.fit_transform(x)
lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly4, y)

plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.title('Linear Regression')
plt.show()

plt.scatter(x, y, color="red")
plt.plot(x, lin_reg2.predict(poly_reg2.fit_transform(x)), color="blue")
plt.title('Polynomial Regression (degree 2)')
plt.show()

plt.scatter(x, y, color="red")
plt.plot(x, lin_reg4.predict(poly_reg4.fit_transform(x)), color="blue")
plt.title('Polynomial Regression (degree 4)')
plt.show()

print("Linear Regression Prediction for 7: ", lin_reg.predict([[7]]))
print("Polynomial Regression (degree 2) Prediction for 7: ", lin_reg2.predict(poly_reg2.fit_transform([[7]])))
print("Polynomial Regression (degree 4) Prediction for 7: ", lin_reg4.predict(poly_reg4.fit_transform([[7]])))

# Decision Tree Regression
veriler = pd.read_csv("veri1.csv")
x = veriler.iloc[:, 1:2].values
y = veriler.iloc[:, 2:].values

dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x, y)

plt.scatter(x, y, color="red")
plt.plot(x, dt_reg.predict(x), color="blue")
plt.title('Decision Tree Regression')
plt.show()

plot_tree(dt_reg, filled=True, rounded=True)
plt.show()

print("Decision Tree Prediction for 5: ", dt_reg.predict([[5]]))

X_grid = np.arange(min(x), max(x) + 0.01, 0.01).reshape(-1, 1)
plt.scatter(x, y, color="blue")
plt.plot(X_grid, dt_reg.predict(X_grid), color="green")
plt.title('Decision Tree Regression (Detailed)')
plt.show()

# Random Forest Regression
veriler = pd.read_csv("veri1.csv")
x = veriler.iloc[:, 1:2].values
y = veriler.iloc[:, 2:].values

rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(x, y.ravel())

plt.scatter(x, y, color="red")
plt.plot(x, rf_reg.predict(x), color="blue")
plt.title('Random Forest Regression')
plt.show()

print("Random Forest Prediction for 6.6: ", rf_reg.predict([[6.6]]))

X_grid = np.arange(min(x), max(x), 0.01).reshape(-1, 1)
plt.scatter(x, y, color='blue')
plt.plot(X_grid, rf_reg.predict(X_grid), color='green')
plt.title('Random Forest Regression (Detailed)')
plt.show()

# Train-test split and model evaluation
data = pd.read_csv("arac_verileri.csv", sep=";")
x = data[['GP', 'DP', 'EP', 'MRGDPI', 'TNL', 'TREP']]
y = data[['BEV']]

x = x.replace(',', '.', regex=True).astype(float)
y = y.replace(',', '.', regex=True).astype(float)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Linear Regression Evaluation
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print("Linear Regression Evaluation")
print("R2: ", r2_score(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))

# Decision Tree Evaluation
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

print("Decision Tree Evaluation")
print("R2: ", r2_score(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))



# Linear Regression for Salary Prediction
# Load the dataset
df = pd.read_csv("D:\\6. semester\\yapay zeka\\2\\linear_regression_dataset.csv", sep=";")

# Reshape the data
x = df.deneyim.values.reshape(-1, 1)
y = df.maas.values.reshape(-1, 1)

# Initialize and train the model
lr = LinearRegression()
lr.fit(x, y)

# Predict the salary for 13 years of experience
predicted_salary = lr.predict([[13]])
print("Predicted salary for 13 years of experience: ", predicted_salary)



# Linear Regression for Sales Prediction
# Load the dataset
veriler = pd.read_csv("D:\\6. semester\\yapay zeka\\2\\satislar.csv")

# Define the independent and dependent variables
x = veriler[['Aylar']]
y = veriler[['Satislar']]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train the model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Make predictions
y_pred = lr.predict(x_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print('R2 Score: ', r2)