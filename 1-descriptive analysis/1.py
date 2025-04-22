import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew,kurtosis
 
data = pd.read_csv("1.csv")
data.columns = data.columns.str.strip()
mean_values = data.mean(numeric_only=True)
median_values = data.median(numeric_only=True)
mode_vales = data.mode(numeric_only=True).iloc[0]

sd = data.std(numeric_only=True)
va = data.var(numeric_only=True)
ran = data.max(numeric_only=True) - data.min(numeric_only=True)
iqr = data.quantile(0.75,numeric_only=True) - data.quantile(0.25,numeric_only=True)

skewness = data.skew(numeric_only=True)
kv = data.kurtosis(numeric_only=True)

print("\nmean\n", mean_values)
print("\nmedian\n",median_values)
print("\nmode\n",mode_vales)
print("\nsd\n",sd)
print("\nva\n",va)
print("\nrange\n",ran)
print("\nIQR\n",iqr)
print("\nskewness\n",skewness)
print("\nkv\n",kv)