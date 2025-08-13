import os
import json
from openai import OpenAI
import api_key  # Load API key from module
from chromadb import PersistentClient


class BookVectorStore:
    def __init__(self, path="./chroma_store", collection_name="book_summaries"):
        self.chroma_client = PersistentClient(path=path)
        print("ChromaDB client initialized.")
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents):
        for i, item in enumerate(documents):
            text = item["summary"] if "summary" in item else str(item)

            embedding = GPTClient.embed_text(text)
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                ids=[f"doc_{i}"]
            )
        print("All summaries stored in ChromaDB.")

    def query(self, query_text, top_k=3):
        embedding = GPTClient.embed_text(query_text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        return results["documents"][0]


class GPTClient:
    api_key = os.getenv("OPENAI_API_KEY")
    #print(f"\napi_key: {api_key}\n")
    client = OpenAI(api_key=api_key)

    @classmethod # accesses class variables
    def embed_text(cls, text):
        response = cls.client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    @classmethod
    def generate_recommendation(cls, query, context_docs):
        context = "\n---\n".join(context_docs)
        prompt = f"""You are a helpful book recommendation assistant.

Based on the user's request and the following book summaries, recommend a book in a warm and conversational tone.

User request: "{query}"

Book summaries:
{context}

Recommendation:"""

        response = cls.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that recommends books based on user preferences."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


class BookRecommenderApp:
    def __init__(self, json_path="file_10_books.json"):
        self.json_path = json_path
        self.query = "Vreau o carte despre prietenie și magie"
        self.vector_store = BookVectorStore()
        print("BookRecommenderApp initialized.")

    def load_summaries(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)
        print(f"Loaded {len(summaries)} summaries")
        return summaries

    def run(self):
        # 1. Load and embed summaries
        summaries = self.load_summaries()
        self.vector_store.add_documents(summaries)

        # 2. Query
        matches = self.vector_store.query(self.query, top_k=3)

        print("\nTop matches:")
        for doc in matches:
            print("-", doc[:100], "...")

        # 3. Generate GPT response
        response = GPTClient.generate_recommendation(self.query, matches)
        print("\nChatGPT response:")
        print(response)


if __name__ == "__main__":
    print("Initializing Book Recommendation System...")
    app = BookRecommenderApp()
    app.run()
