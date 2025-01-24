import torch
from torch.optim import Adam
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Step 1: Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token for GPT-2

# Load the preprocessed data from CSV files
go_emotions_train_df = pd.read_csv("go_emotions_preprocessed.csv")
daily_dialog_train_df = pd.read_csv("daily_dialog_preprocessed.csv")
go_emotions_valid_df = pd.read_csv("go_emotions_preprocessed.csv")  # Assuming validation data is also in the same CSV
daily_dialog_valid_df = pd.read_csv("daily_dialog_preprocessed.csv")  # Assuming validation data is also in the same CSV

# Step 2: Ensure that input_ids and attention_mask are lists of integers
def convert_to_list_of_ints(df, column_name):
    return [eval(item) if isinstance(item, str) else item for item in df[column_name]]

# Convert the string representations of lists into actual lists of integers
go_emotions_train_df['input_ids'] = convert_to_list_of_ints(go_emotions_train_df, 'input_ids')
go_emotions_train_df['attention_mask'] = convert_to_list_of_ints(go_emotions_train_df, 'attention_mask')
go_emotions_valid_df['input_ids'] = convert_to_list_of_ints(go_emotions_valid_df, 'input_ids')
go_emotions_valid_df['attention_mask'] = convert_to_list_of_ints(go_emotions_valid_df, 'attention_mask')

daily_dialog_train_df['input_ids'] = convert_to_list_of_ints(daily_dialog_train_df, 'input_ids')
daily_dialog_train_df['attention_mask'] = convert_to_list_of_ints(daily_dialog_train_df, 'attention_mask')
daily_dialog_valid_df['input_ids'] = convert_to_list_of_ints(daily_dialog_valid_df, 'input_ids')
daily_dialog_valid_df['attention_mask'] = convert_to_list_of_ints(daily_dialog_valid_df, 'attention_mask')

# Step 3: Create a custom Dataset for PyTorch DataLoader
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# For GoEmotions dataset (train and validation)
go_emotions_train_encodings = {
    'input_ids': go_emotions_train_df['input_ids'].tolist(),
    'attention_mask': go_emotions_train_df['attention_mask'].tolist()
}
go_emotions_train_labels = go_emotions_train_df['labels'].tolist()
go_emotions_valid_encodings = {
    'input_ids': go_emotions_valid_df['input_ids'].tolist(),
    'attention_mask': go_emotions_valid_df['attention_mask'].tolist()
}
go_emotions_valid_labels = go_emotions_valid_df['labels'].tolist()

# For DailyDialog dataset (train and validation)
daily_dialog_train_encodings = {
    'input_ids': daily_dialog_train_df['input_ids'].tolist(),
    'attention_mask': daily_dialog_train_df['attention_mask'].tolist()
}
daily_dialog_train_labels = daily_dialog_train_df['labels'].tolist()
daily_dialog_valid_encodings = {
    'input_ids': daily_dialog_valid_df['input_ids'].tolist(),
    'attention_mask': daily_dialog_valid_df['attention_mask'].tolist()
}
daily_dialog_valid_labels = daily_dialog_valid_df['labels'].tolist()

# Step 4: Create datasets and dataloaders
train_dataset = EmotionDataset(daily_dialog_train_encodings, daily_dialog_train_labels)  # or go_emotions_train_encodings
val_dataset = EmotionDataset(daily_dialog_valid_encodings, daily_dialog_valid_labels)  # or go_emotions_valid_encodings

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4)

# Step 5: Set up optimizer and loss function
optimizer = Adam(model.parameters(), lr=2e-5)

# Step 6: Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = {key: val.to(device) for key, val in batch.items()}

        # Forward pass
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])

        # Calculate loss
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Training loss: {avg_train_loss}")

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
            loss = outputs.loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation loss: {avg_val_loss}")

# Step 7: Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

print("Training complete and model saved.")
