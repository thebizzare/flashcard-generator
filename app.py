import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Get Hugging Face token
token = os.getenv("HF_TOKEN")

# Create Hugging Face client
client = InferenceClient(
    api_key=token
)

# Select model
model = "Qwen/Qwen3-4B-Instruct-2507"

# Take input from user
topic = input("Enter the topic: ")
number = input("How many flashcards do you want? ")

# Create prompt
prompt = f"""
Create {number} flashcards about {topic}.

For each flashcard, provide:

Q: Question
A: Answer

Keep the questions simple, clear, and educational.
"""

# Send request to Hugging Face model
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# Extract generated text
flashcards = response.choices[0].message.content

# Display result
print("\nGenerated Flashcards:\n")
print(flashcards)