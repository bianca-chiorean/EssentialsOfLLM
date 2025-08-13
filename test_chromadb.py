import os
import json
from openai import OpenAI
import api_key  # Load API key from module
from chromadb import PersistentClient
from openai.types.chat import ChatCompletionMessageToolCall

# Dictionary of full summaries
book_summaries_dict = {
    "The Hobbit": (
        "Bilbo Baggins, un hobbit confortabil și fără aventuri, este luat prin surprindere "
        "atunci când este invitat într-o misiune de a recupera comoara piticilor păzită de dragonul Smaug. "
        "Pe parcursul călătoriei, el descoperă curajul și resursele interioare pe care nu știa că le are. "
        "Povestea este plină de creaturi fantastice, prietenii neașteptate și momente tensionate."
    ),
    "1984": (
        "Romanul lui George Orwell descrie o societate distopică aflată sub controlul total al statului. "
        "Oamenii sunt supravegheați constant de „Big Brother”, iar gândirea liberă este considerată crimă. "
        "Winston Smith, personajul principal, încearcă să reziste acestui regim opresiv. "
        "Este o poveste despre libertate, adevăr și manipulare ideologică."
    )
}


# Tool function
def get_summary_by_title(title: str) -> str:
    return book_summaries_dict.get(title, "Titlul nu a fost găsit în baza de date.")

# Tool schema for OpenAI function calling
function_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Returnează rezumatul complet al unei cărți după titlu exact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Titlul cărții (exact)"
                    }
                },
                "required": ["title"]
            }
        }
    }
]


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
        # response = GPTClient.generate_recommendation(self.query, matches)
        # print("\nChatGPT response:")
        # print(response)
        # Prepare the prompt and context
        context = "\n---\n".join(matches)
        prompt = f"""Recomandă o carte potrivită bazat pe rezumatul acestor cărți.

        Întrebarea utilizatorului: "{self.query}"

        Rezumate:
        {context}

        Recomandă titlul (exact), apoi apelează tool-ul pentru a oferi rezumatul complet.
        """

        # Initial GPT call with tool available
        chat_response = GPTClient.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Ești un asistent prietenos care recomandă cărți."},
                {"role": "user", "content": prompt}
            ],
            tools=function_tools,
            tool_choice="auto"
        )

        # Show GPT reply
        message = chat_response.choices[0].message
        print("\nChatGPT recommendation:")
        print(message.content or "")

        # Check for tool call
        tool_calls = message.tool_calls
        if tool_calls:
            print("\nTool calls detected:")
            for call in tool_calls:
                if call.function.name == "get_summary_by_title":
                    args = json.loads(call.function.arguments)
                    title = args["title"]

                    summary = get_summary_by_title(title)
                    print(f"\nRezumat complet pentru '{title}':")
                    print(summary)



if __name__ == "__main__":
    print("Initializing Book Recommendation System...")
    app = BookRecommenderApp()
    app.run()
