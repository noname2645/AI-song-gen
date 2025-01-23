import pandas as pd
import re
from datasets import Dataset

# Step 1: Load CSV file
file_path = (r"D:\AI song generation\AI\scrapped-lyrics-from-6-genres\lyrics-data.csv")
file1 = pd.read_csv(file_path)

# Rename columns for consistency
file1 = file1.rename(columns={"Lyric": "text"})  # Rename 'lyrics' to 'text'

# Step 2: Filter and clean text data
file1 = file1.dropna(subset=['text'])  # Remove rows with NaN in 'text'
file1['text'] = file1['text'].astype(str)  # Ensure all values in 'text' are strings

# Clean text function
def clean_text(text):
    """Clean the text by removing special characters and unnecessary whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s.,!?\'"]+', '', text)  # Remove special characters except common punctuation
    return text.strip()

# Apply text cleaning
file1['text'] = file1['text'].apply(clean_text)

# Step 3: Prepare Hugging Face Dataset
dataset = Dataset.from_pandas(file1[['text']])  # Only include 'text' column

# Split dataset into train and test sets (if not already split)
dataset = dataset.train_test_split(test_size=0.1)

# Save the preprocessed dataset to a file for later use
dataset.save_to_disk("preprocessed_dataset")
print("Preprocessed dataset saved.")
