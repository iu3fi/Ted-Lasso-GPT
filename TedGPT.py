# Install necessary libraries
!pip install transformers datasets --quiet

# Import necessary libraries
import pandas as pd
from google.colab import drive
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Mount Google Drive
drive.mount('/content/drive')

# Load dataset
file_path = '/content/drive/MyDrive/Dataset.csv'  # Update path if necessary
data = pd.read_csv(file_path)

# Preprocessing the dataset
# Display the first few rows, check for null values and column names
print(data.head())
print(data.isnull().sum())
print(data.columns)

# Rename columns if necessary
data.rename(columns={'questions': 'input', 'answers': 'output'}, inplace=True)

# Drop rows with null values and strip extra spaces
data.dropna(subset=['input', 'output'], inplace=True)
data['input'] = data['input'].str.strip()
data['output'] = data['output'].str.strip()

# Display a sample after preprocessing
print(data.head())

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Reset indices for both datasets
train_data.reset_index(drop=True, inplace=True)
val_data.reset_index(drop=True, inplace=True)

# Display dataset sizes
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

# Convert pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# Verify the dataset structure
print(train_dataset[0])  # Display a sample from the training dataset

# Load the tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the pad_token to be the same as eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define the tokenization function
def tokenize_function(examples):
    inputs = examples['input']
    outputs = examples['output']
    concatenated = [f"{inp} {tokenizer.eos_token} {out}" for inp, out in zip(inputs, outputs)]
    return tokenizer(concatenated, truncation=True, padding='max_length', max_length=512)

# Apply tokenization to both train and validation sets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Format dataset for causal language modeling
def format_dataset(examples):
    examples['labels'] = examples['input_ids'].copy()  # Labels are the same as input_ids for causal LM
    return examples

formatted_train = tokenized_train.map(format_dataset, batched=True)
formatted_val = tokenized_val.map(format_dataset, batched=True)

# Set the format to PyTorch tensors
formatted_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
formatted_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=formatted_train,
    eval_dataset=formatted_val,
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model('/content/drive/MyDrive/TLDS_FineTuned')
tokenizer.save_pretrained('/content/drive/MyDrive/TLDS_FineTuned')

# Load the fine-tuned model and tokenizer for inference
fine_tuned_model = AutoModelForCausalLM.from_pretrained('/content/drive/MyDrive/TLDS_FineTuned')
fine_tuned_tokenizer = AutoTokenizer.from_pretrained('/content/drive/MyDrive/TLDS_FineTuned')

# Function to generate a response
def generate_response(prompt, max_length=100):
    inputs = fine_tuned_tokenizer.encode(prompt + fine_tuned_tokenizer.eos_token, return_tensors='pt')
    outputs = fine_tuned_model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=fine_tuned_tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    response = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# Interactive chat
print("Welcome to Ted Lasso Bot! Type 'exit' to end the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Ted Lasso Bot: Take care! You're amazing!")
        break
    response = generate_response(user_input)
    print("Ted Lasso Bot:", response)
