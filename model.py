import pandas as pd
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("laptop_data.csv")

X = data[["RAM_GB", "SSD_GB", "Weight_kg"]]
y = data["Price"]

model = LinearRegression()
model.fit(X, y)

print("Model trained successfully")