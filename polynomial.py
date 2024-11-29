import main as m
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from scipy.stats import zscore
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

x_small = m.data.iloc[:100, 6:15]
spectrum_filtered_small = pd.DataFrame(savgol_filter(x_small, 7, 3, deriv = 2, axis = 0))
spectrum_filtered_st_small = zscore(spectrum_filtered_small, axis = 1)

X = spectrum_filtered_st_small
y = m.data['PURITY'][:100]
X_train, X_valid, y_train , y_valid = train_test_split(X, y, test_size=0.05, random_state=42) 
print(x_small.shape)
print(y.shape)

# Create a Polynomial Regression model
model3 = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
param_grid = {'polynomialfeatures__degree': range(1, 9)}
#grid_search = GridSearchCV(model3, param_grid, cv=10, scoring='neg_mean_squared_error')
random_search = RandomizedSearchCV(estimator=model3, param_distributions=param_grid, n_iter=8, cv=5)
random_search.fit(m.X_train, m.y_train)

#grid_search.fit(m.X_train, m.y_train)
best_degree = random_search.best_params_['polynomialfeatures__degree']

best_model3 = make_pipeline(PolynomialFeatures(degree=best_degree), 
                            LinearRegression())
best_model3.fit(m.X_train, m.y_train)

y_pred3 = best_model3.predict(m.X_test)
#rmse_poly = mean_squared_error(m.y_test, y_pred3, squared=False)
#f'RMSE for best polynomial regressor: {rmse_poly:.3g}, with degree={best_degree}'
t_score = np.mean(np.abs(m.y_pred - m.y_valid) <= 5)
print(t_score)




