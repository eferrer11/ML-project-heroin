import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import zscore

data = pd.read_csv("train.csv")
spectrum = data.iloc[:, 6:]
spectrum_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0))
spectrum_filtered_standardized = zscore(spectrum_filtered, axis = 1)
