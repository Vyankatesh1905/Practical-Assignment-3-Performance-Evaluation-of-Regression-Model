import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

print("First five records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

X = df.drop(columns=["target"])
y = df["target"]

print("\nInput Features:")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

def evaluate_model(model_name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    print("\n--------------------------------")
    print(model_name)
    print("--------------------------------")
    print("MAE  :", round(mae, 4))
    print("MSE  :", round(mse, 4))
    print("RMSE :", round(rmse, 4))
    print("R2   :", round(r2, 4))

    return mae, mse, rmse, r2

linear_metrics = evaluate_model(
    "Multiple Linear Regression",
    y_test,
    linear_pred
)

ridge_metrics = evaluate_model(
    "Ridge Regression",
    y_test,
    ridge_pred
)

bmi = df["bmi"].values

bmi_train, bmi_test, target_train, target_test = train_test_split(
    bmi, y.values, test_size=0.20, random_state=42
)

bmi_mean = bmi_train.mean()
bmi_std = bmi_train.std()

bmi_train_scaled = (bmi_train - bmi_mean) / bmi_std
bmi_test_scaled = (bmi_test - bmi_mean) / bmi_std

slope = 0.0
intercept = 0.0

learning_rate = 0.05
iterations = 2000

n = len(bmi_train_scaled)
cost_history = []

for iteration in range(iterations):

    prediction = slope * bmi_train_scaled + intercept
    error = prediction - target_train

    slope_gradient = (2 / n) * np.sum(error * bmi_train_scaled)
    intercept_gradient = (2 / n) * np.sum(error)

    slope -= learning_rate * slope_gradient
    intercept -= learning_rate * intercept_gradient

    cost = np.mean(
        (target_train - (slope * bmi_train_scaled + intercept)) ** 2
    )

    cost_history.append(cost)

gd_predictions = slope * bmi_test_scaled + intercept

gd_metrics = evaluate_model(
    "Gradient Descent Linear Regression",
    target_test,
    gd_predictions
)

print("\nGradient Descent Parameters")
print("----------------------------")
print("Slope:", round(slope, 4))
print("Intercept:", round(intercept, 4))
print("Final Training MSE:", round(cost_history[-1], 4))

bmi_lr = LinearRegression()

bmi_lr.fit(
    bmi_train.reshape(-1, 1),
    target_train
)

bmi_lr_prediction = bmi_lr.predict(
    bmi_test.reshape(-1, 1)
)

bmi_lr_metrics = evaluate_model(
    "Scikit-learn BMI Linear Regression",
    target_test,
    bmi_lr_prediction
)

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    linear_pred,
    label="Predicted Values"
)

minimum = min(y_test.min(), linear_pred.min())
maximum = max(y_test.max(), linear_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Disease-Progression Value")
plt.ylabel("Predicted Disease-Progression Value")
plt.title("Actual vs Predicted Disease-Progression Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))

plt.plot(
    range(1, iterations + 1),
    cost_history
)

plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Cost Reduction")
plt.grid(True)
plt.tight_layout()
plt.show()

model_names = [
    "Linear Regression",
    "Ridge Regression"
]

r2_values = [
    linear_metrics[3],
    ridge_metrics[3]
]

plt.figure(figsize=(7, 5))

plt.bar(
    model_names,
    r2_values
)

plt.ylabel("R² Score")
plt.title("R² Score Comparison of Regression Models")
plt.grid(
    axis="y",
    linestyle="--"
)
plt.tight_layout()
plt.show()

example_bmi = 0.0

example_scaled = (example_bmi - bmi_mean) / bmi_std

example_prediction = slope * example_scaled + intercept

print("\nExample BMI Prediction")
print("----------------------")
print("Standardized BMI:", example_bmi)
print(
    "Predicted disease-progression value:",
    round(example_prediction, 2)
)