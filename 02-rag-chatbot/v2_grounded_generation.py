import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

document = """
Refund Policy
Customers can request refunds within 30 days.

Work From Home Policy
Employees may work remotely three days per week.

Vacation Policy
Employees receive 20 days of paid leave.
"""

def load_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API Key not found! Please check your .env file.")
    return api_key

def create_client(api_key):
    client = genai.Client(api_key=api_key)
    return client

def build_prompt(query, context):
    return f"""
        You are a helpful assistant.

        Answer ONLY using the provided context.

        If the answer is not present in the context,
        say "I don't know."

        Context:
        {context}

        Question:
        {query}"""

def send_prompt(client, model, prompt):
    response = client.models.generate_content(
        model= model,
        contents= prompt
        )

    return response

def retrieve(model, query, chunks, chunk_embeddings):
    embeded_query = model.encode(query)
    best_score = 0
    best_index = 0
    for i in range(len(chunks)):
        similarity = cosine_similarity([embeded_query], [chunk_embeddings[i]])
        if best_score<similarity.item():
            best_score = similarity
            best_index = i
    return chunks[best_index]


def main():
    print("started script")
    chunks = document.split("\n\n")
    
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_model = "gemini-2.5-flash"
    client = create_client(load_api_key())
    print("\n\nstarted embedding chunks\n\n")
    chunk_embeddings = emb_model.encode(chunks)

    query = "What is the refund policy?"

    best_chunk = retrieve(model=emb_model, query=query,chunks=chunks, chunk_embeddings=chunk_embeddings)
    
    retrieve_prompt = build_prompt(query=query, context=best_chunk)

    response = send_prompt(client=client, model=gen_model, prompt=retrieve_prompt)

    print(response.text)

if __name__ == "__main__":
    main()