import os
from openai import OpenAI
import api_key  # Import the API key from api_key.py
import json
import chromadb
from chromadb import PersistentClient


api_key = os.getenv("OPENAI_API_KEY")
#print("Loaded key:", api_key)

client = OpenAI(api_key=api_key)  # Will use OPENAI_API_KEY from environment

print("OpenAI API key loaded.")

# Initialize ChromaDB client
chroma_client = PersistentClient(path="./chroma_store")
print("ChromaDB client initialized.")

# Create or get a collection
collection = chroma_client.get_or_create_collection(name="book_summaries")

# Load summaries from file
with open("file_10_books.json", "r", encoding="utf-8") as f:
    summaries = json.load(f)

print(f"Loaded {len(summaries)} summaries")

# Process each summary
for i, item in enumerate(summaries):
    text = item["summary"] if "summary" in item else str(item)  # Adjust to your JSON structure

    # Create embedding
    embedding = client.embeddings.create( # calls OpenAI API to generate an embedding
        input=[text], # input text to be embedded
        model="text-embedding-3-small" # specify the embedding model
    ).data[0].embedding # extract the embedding vector
    
    # Add to ChromaDB
    collection.add(
        embeddings=[embedding], #embedding vector
        documents=[text], # original text
        ids=[f"doc_{i}"] # id
    )

print("All summaries stored in ChromaDB.")


# Seach:
query = "Vreau o carte despre prietenie și magie" #"exploration of identity in science fiction"

# Use same embedding model
query_embedding = client.embeddings.create(
    input=[query],
    model="text-embedding-3-small"
).data[0].embedding

# Perform semantic search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3  # return top 3 most similar
)

print("Top matches:")
for doc in results["documents"][0]:
    print("-", doc[:100], "...")


# Join all documents into one string with separators
context = "\n---\n".join(results["documents"][0])

# Craft your full prompt
prompt = f"""You are a helpful book recommendation assistant.

Based on the user's request and the following book summaries, recommend a book in a warm and conversational tone.

User request: "{query}"

Book summaries:
{context}

Recommendation:"""

chat_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that recommends books based on user preferences."},
        {"role": "user", "content": prompt}
    ]
)
print("\nChatGPT response:")
print(chat_response.choices[0].message.content)
