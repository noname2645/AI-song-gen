import pandas as pd

# Load the dataset
df = pd.read_csv(r"D:\AI song generation\AI\scrapped-lyrics-from-6-genres\lyrics-data.csv")

# Check column names
print(df.columns)

column_names = df.columns.tolist()
print(column_names)
