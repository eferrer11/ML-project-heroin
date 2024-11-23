import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("./train.csv")
spectrum = data.iloc[:, 6:]
spectrum_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0))
spectrum_filtered_st = zscore(spectrum_filtered, axis = 1)

test_data = pd.read_csv("./test.csv")
spectrum_test = test_data.iloc[:, 6:]
spectrum_test_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0))
spectrum_test_filtered_st = zscore(spectrum_filtered, axis = 1)

model = LinearRegression()
X = spectrum_filtered_st
y = data['PURITY']
X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.05, random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)

print("Nombre de prédicteurs :", X.shape[1])