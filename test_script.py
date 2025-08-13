import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import api_key  # Import the API key from api_key.py
import json

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)  # Will use OPENAI_API_KEY from environment

print("OpenAI API key loaded.")

# Test embedding
text = "A test string about adventure and discovery."

response = client.embeddings.create(
    input=[text],
    model="text-embedding-3-small"
)

embedding = response.data[0].embedding

print("Embedding generated.")
print("First 5 values of the embedding vector:", embedding[:5])


#test summary file
with open("file_10_books.json", "r", encoding="utf-8") as f:
    summaries = json.load(f)

print("Loaded", len(summaries), "summaries")