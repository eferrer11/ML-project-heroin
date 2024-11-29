import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV 
from sklearn.linear_model import Lasso, Ridge
from scipy.spatial.distance import cdist

# Load the data
data = pd.read_csv("./train.csv")
spectrum = data.iloc[:, 6:]
spectrum_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0))
spectrum_filtered_st = zscore(spectrum_filtered, axis = 1)


# Code for the new features
substances_pure = pd.read_csv("./substances.csv")
pure_heroin = substances_pure[(substances_pure["substance"] == "heroin (white)") | (substances_pure["substance"] == "heroin (brown)")]
data_new_features = cdist(data.iloc[:,6:].to_numpy(), pure_heroin.iloc[:,1:].to_numpy(), metric='euclidean')
data_new_features_df = pd.DataFrame(data_new_features, index=data.iloc[:,6:].index, columns=pure_heroin.iloc[:,1:].index)


# Save the data with new features and filtered spectrum
data_features = pd.concat([data, data_new_features_df], axis=1)
data_features.to_csv('data_features.csv', index=False)
spectrum_new_features = data_new_features_df.iloc[:, 6:]
spectrum_filtered_new_features = pd.DataFrame(savgol_filter(spectrum_new_features, 7, 3, deriv = 2, axis = 0))
spectrum_filtered_st_new_features = zscore(spectrum_filtered_new_features, axis = 1)

"""
test_data = pd.read_csv("./test.csv")
spectrum_test = test_data.iloc[:, 6:]
spectrum_test_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0))
spectrum_test_filtered_st = zscore(spectrum_filtered, axis = 1) 
"""
"""
correlation  = data.iloc[:, 6:].corr()
np.fill_diagonal(correlation, 0)
df_clean = spectrum_filtered_st.drop(spectrum_filtered_st.columns(np.where(correlation >= 0.99), axis = 1))
"""
# suppress the columns with a correlation > 0.99
corrwithpurity = data.iloc[:, 6:].corrwith(data['PURITY'])
spectrum_filtered_st = spectrum_filtered_st.drop(spectrum_filtered_st.columns[np.where(np.abs(corrwithpurity) <= 0.3)[0]], axis=1)

def tune_model(model, data, target):

    # Create a KFold cross-validator with 20 folds
    kf = KFold(n_splits=20, shuffle=True, random_state=42)

    # Define the parameter grid for tuning alpha
    param_grid = {
        'alpha': np.logspace(-30, -1, num =30)
    }

    # Create a grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_root_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(target, data)

    return grid_search


#Linear Regression for the filtered spectrum with lasso
model = LinearRegression()
X = spectrum_filtered_st
y = data['PURITY']
X_train, X_valid, y_train , y_valid = train_test_split(X, y, test_size=0.05, random_state=42) 
model.fit(X_train,y_train)
y_pred = model.predict(X_valid)
#t_score = np.mean(np.abs(res1_lasso - y_valid) <= 5)
#print(t_score)

#Linear Regression for the filtered spectrum with new features
model = LinearRegression()
X = spectrum_filtered_st_new_features
y = data['PURITY']
X_train, X_valid, y_train , y_valid = train_test_split(X, y, test_size=0.05, random_state=42) 
model.fit(X_train,y_train)
y_pred = model.predict(X_valid)
t_score = np.mean(np.abs(y_pred - y_valid) <= 5)
print('t_score without new features:', t_score)

#Linear Regression for the filtered spectrum with new features
model = LinearRegression()
X = spectrum_filtered_st
y = data['PURITY']
X_train, X_valid, y_train , y_valid = train_test_split(X, y, test_size=0.05, random_state=42) 
model.fit(X_train,y_train)
y_pred = model.predict(X_valid)
t_score = np.mean(np.abs(y_pred - y_valid) <= 5)
print('t_score with new features:', t_score)