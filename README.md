# Ted Lasso Pep Talk Chatbot

This project implements a fine-tuned GPT-2 model trained on Ted Lasso script dialogues to create a chatbot that delivers uplifting and motivational pep talks. The chatbot takes user input and responds with an encouraging message, just like Ted Lasso himself!

---

## Features
- Fine-tuned GPT-2 model for conversational responses.
- Interactive chat interface for seamless user interaction.
- Supports input prompts and delivers motivational responses.
- Can be run locally using Google Colab.

---

## Dataset
The dataset consists of **500 Q&A pairs** based on the Ted Lasso series. Each pair contains:
- **Input**: A user question or statement.
- **Output**: Ted Lasso-style motivational responses.

---

## How It Works
1. The GPT-2 model is fine-tuned on the dataset of Q&A pairs.
2. The chatbot generates contextually relevant motivational responses.
3. The responses are personalized based on the input text.

---

## Installation
To run the chatbot, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ted-lasso-chatbot.git
cd ted-lasso-chatbot
```

### 2. Install Dependencies
Install the required Python libraries:
```bash
pip install transformers datasets torch
```

### 3. Download or Prepare the Dataset
Place the CSV dataset (`Dataset.csv`) in the project directory. Ensure it has the following columns:
- `questions`: User questions or inputs.
- `answers`: Ted Lasso-style responses.

---

## Usage

### 1. Train the Model
To fine-tune the GPT-2 model:
1. Open `TedGPT.py` in Google Colab.
2. Follow the step-by-step instructions in the notebook to fine-tune the model.
3. Save the fine-tuned model to Google Drive or your local machine.

### 2. Run the Chatbot
Use the following Python script to interact with the chatbot:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model
model_path = "path_to_your_finetuned_model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Chat function
def chat_with_ted():
    print("Welcome to Ted Lasso Bot! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ted Lasso Bot: Take care! You're amazing!")
            break
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        outputs = model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Ted Lasso Bot:", response)

# Start the chat
chat_with_ted()
```
Replace `path_to_your_finetuned_model` with the path to your saved model.

---

## Example Interaction
```
Welcome to Ted Lasso Bot! Type 'exit' to end the chat.
You: I'm feeling really down today.
Ted Lasso Bot: You got this! Believe in yourself, one small step at a time.
You: How do I handle failure?
Ted Lasso Bot: Failure is just an opportunity to learn and grow. Chin up!
You: exit
Ted Lasso Bot: Take care! You're amazing!
```

---

## Project Structure
```
├── Dataset.csv                # Dataset file
├── TedGPT.py     # Python script for running the chatbot
├── README.md               # Project documentation
```

---

## Future Improvements
- Add more Q&A pairs to enhance the chatbot's conversational abilities.
- Incorporate sentiment analysis to detect user mood and provide more tailored responses.
- Deploy the chatbot on a web interface using Flask or Streamlit.

---

## Acknowledgments
- [Hugging Face](https://huggingface.co/) for the transformers library.
- Ted Lasso creators for the inspiring script dialogues.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
