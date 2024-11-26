import main as m

# Create a Polynomial Regression model
model3 = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
param_grid = {'polynomialfeatures__degree': range(1, 9)}
grid_search = GridSearchCV(model3, param_grid, cv=10,           
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_degree = grid_search.best_params_['polynomialfeatures__degree']

best_model3 = make_pipeline(PolynomialFeatures(degree=best_degree), 
                            LinearRegression())
best_model3.fit(X_train, y_train)

y_pred3 = best_model3.predict(X_test)
rmse_poly = mean_squared_error(y_test, y_pred3, squared=False)
f'RMSE for best polynomial regressor: {rmse_poly:.3g}, with degree={best_degree}'