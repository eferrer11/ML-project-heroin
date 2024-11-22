import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import zscore

#train
data = pd.read_csv("train.csv")
spectrum = data.iloc[:, 6:]
spectrum_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0))
spectrum_filtered_standardized = zscore(spectrum_filtered, axis = 1)

#test
data = pd.read_csv("test.csv")
spectrum_test = data.iloc[:, 6:]
spectrum_filtered_test = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0))
spectrum_filtered_standardized_test = zscore(spectrum_filtered, axis = 1)
