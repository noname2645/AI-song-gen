import torch
from torch.optim import Adam
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader

# Step 1: Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token for GPT-2

# Step 2: Define dataset and dataloaders (using raw text)
# Assuming your dataset is already in raw form, a simple list of text examples
train_texts = ["Your raw song lyrics 1", "Your raw song lyrics 2", "Your raw song lyrics 3"]
val_texts = ["Validation raw song lyrics 1", "Validation raw song lyrics 2"]

# Tokenize directly inside the DataLoader
def encode_text(text):
    return tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

train_encodings = [encode_text(text) for text in train_texts]
val_encodings = [encode_text(text) for text in val_texts]

# Step 3: Create DataLoader for batching (using encoded inputs)
train_dataloader = DataLoader(train_encodings, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_encodings, batch_size=4)

# Step 4: Set up optimizer and loss function
optimizer = Adam(model.parameters(), lr=2e-5)

# Step 5: Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        # Move batch to device
        inputs = {key: val.to(device) for key, val in batch.items()}

        # Forward pass (labels are input_ids for GPT-2)
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["input_ids"])

        # Calculate loss (language model loss)
        loss = outputs.loss
        if loss is not None:  # Check if loss is not None
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
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["input_ids"])
            loss = outputs.loss
            if loss is not None:  # Check if loss is not None
                total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation loss: {avg_val_loss}")

# Step 6: Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

print("Training complete and model saved.")
