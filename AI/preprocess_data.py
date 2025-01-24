from transformers import BertTokenizer
from datasets import load_dataset
import torch
import pandas as pd

# Function to load and preprocess the datasets
def preprocess_datasets():
    # Load the datasets
    go_emotions_dataset = load_dataset("go_emotions")
    dailydialog_dataset = load_dataset("daily_dialog", trust_remote_code=True)
    
    # Load pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenization function
    def tokenize_function(examples, dataset_name):
        if dataset_name == "go_emotions":
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
        elif dataset_name == "daily_dialog":
            # Concatenate all sentences in the 'dialog' list into a single string
            dialogues = [' '.join(dialog) for dialog in examples['dialog']]
            return tokenizer(dialogues, padding="max_length", truncation=True, max_length=128)

    # Tokenize GoEmotions dataset
    go_emotions_tokenized = go_emotions_dataset.map(lambda examples: tokenize_function(examples, "go_emotions"), batched=True)
    go_emotions_train_dataset = go_emotions_tokenized["train"]
    go_emotions_valid_dataset = go_emotions_tokenized["validation"]
    
    # Tokenize DailyDialog dataset
    dailydialog_tokenized = dailydialog_dataset.map(lambda examples: tokenize_function(examples, "daily_dialog"), batched=True)
    dailydialog_train_dataset = dailydialog_tokenized["train"]
    dailydialog_valid_dataset = dailydialog_tokenized["validation"]

    # Process labels for GoEmotions
    go_emotions_train_labels = torch.tensor([example['labels'][0] for example in go_emotions_train_dataset], dtype=torch.long)
    go_emotions_valid_labels = torch.tensor([example['labels'][0] for example in go_emotions_valid_dataset], dtype=torch.long)

    # Handle DailyDialog Emotion labels: Use the first emotion from the list for each dialogue
    dailydialog_train_labels = torch.tensor([example['emotion'][0] for example in dailydialog_train_dataset], dtype=torch.long)
    dailydialog_valid_labels = torch.tensor([example['emotion'][0] for example in dailydialog_valid_dataset], dtype=torch.long)

    # Return tokenized datasets and labels
    return {
        "go_emotions": {
            "train": go_emotions_train_dataset,
            "valid": go_emotions_valid_dataset,
            "train_labels": go_emotions_train_labels,
            "valid_labels": go_emotions_valid_labels
        },
        "daily_dialog": {
            "train": dailydialog_train_dataset,
            "valid": dailydialog_valid_dataset,
            "train_labels": dailydialog_train_labels,
            "valid_labels": dailydialog_valid_labels
        }
    }

# Run the preprocessing
preprocessed_data = preprocess_datasets()

# Convert GoEmotions to DataFrame
go_emotions_train_df = pd.DataFrame({
    'input_ids': [item['input_ids'] for item in preprocessed_data["go_emotions"]["train"]],
    'attention_mask': [item['attention_mask'] for item in preprocessed_data["go_emotions"]["train"]],
    'labels': preprocessed_data["go_emotions"]["train_labels"].tolist()
})

# Convert DailyDialog to DataFrame
dailydialog_train_df = pd.DataFrame({
    'input_ids': [item['input_ids'] for item in preprocessed_data["daily_dialog"]["train"]],
    'attention_mask': [item['attention_mask'] for item in preprocessed_data["daily_dialog"]["train"]],
    'labels': preprocessed_data["daily_dialog"]["train_labels"].tolist()
})

# Save the data to CSV
go_emotions_train_df.to_csv("go_emotions_preprocessed.csv", index=False)
dailydialog_train_df.to_csv("daily_dialog_preprocessed.csv", index=False)

print("Data saved as CSV successfully.")

# Example of accessing the preprocessed data for GoEmotions
print("GoEmotions Example:")
print(preprocessed_data["go_emotions"]["train"][0])

# Example of accessing the preprocessed data for DailyDialog
print("\nDailyDialog Example:")
print(preprocessed_data["daily_dialog"]["train"][0])
