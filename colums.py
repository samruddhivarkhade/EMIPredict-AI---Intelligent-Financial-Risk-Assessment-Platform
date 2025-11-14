import pandas as pd
data = pd.read_csv("artifacts/processed_data.csv")
print(data.columns.tolist())
