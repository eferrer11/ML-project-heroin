import main as m
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV

# Create a Polynomial Regression model
model3 = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
param_grid = {'polynomialfeatures__degree': range(1, 9)}
grid_search = GridSearchCV(model3, param_grid, cv=10,           
                           scoring='neg_mean_squared_error')
grid_search.fit(m.X_train, m.y_train)
best_degree = grid_search.best_params_['polynomialfeatures__degree']

best_model3 = make_pipeline(PolynomialFeatures(degree=best_degree), 
                            LinearRegression())
best_model3.fit(m.X_train, m.y_train)

y_pred3 = best_model3.predict(m.X_test)
rmse_poly = mean_squared_error(m.y_test, y_pred3, squared=False)
f'RMSE for best polynomial regressor: {rmse_poly:.3g}, with degree={best_degree}'