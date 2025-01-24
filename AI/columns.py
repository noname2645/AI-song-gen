# Import necessary libraries
import pandas as pd

# Load the CSV files into pandas DataFrames
go_emotions_train_df = pd.read_csv('go_emotions_preprocessed.csv')
daily_dialog_train_df = pd.read_csv('daily_dialog_preprocessed.csv')

# Print the columns of the dataframes to inspect them
print("GoEmotions columns:")
print(go_emotions_train_df.columns)

print("\nDailyDialog columns:")
print(daily_dialog_train_df.columns)


