import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

document = """
Refund Policy
Customers can request refunds within 30 days of purchase. Refunds are processed within 5 business days after approval.

Work From Home Policy
Employees may work remotely up to three days per week with manager approval.

Vacation Policy
Employees receive 20 days of paid leave annually. Unused leave can be carried forward for up to 5 days.

Health Insurance Policy
The company provides medical insurance coverage for employees and their dependents.

Travel Reimbursement Policy
Business travel expenses including flights, hotels, and meals will be reimbursed upon submission of receipts.

Laptop Policy
Every employee is assigned a company laptop. Employees are responsible for maintaining and securing company devices.

Training Policy
Employees are entitled to an annual training budget of $1000 for professional development courses and certifications.

Performance Review Policy
Performance reviews are conducted twice a year, in June and December.

Dress Code Policy
Employees are expected to maintain business casual attire while working from the office.

Office Timing Policy
Regular office hours are from 9 AM to 6 PM, Monday through Friday.

Resignation Policy
Employees must provide a minimum notice period of 30 days before leaving the company.

Parental Leave Policy
Employees are entitled to 12 weeks of paid parental leave following the birth or adoption of a child.
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
    threshold = 0.3
    for i in range(len(chunks)):
        similarity = cosine_similarity([embeded_query], [chunk_embeddings[i]]).item()
        if best_score<similarity:
            best_score = similarity
            best_index = i
    if best_score < threshold:
        return "No relevent documents found"
    else:
        return {"chunk" : chunks[best_index],
                "score" : best_score}


def main():
    print("started script")
    chunks = document.split("\n\n")
    
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_model = "gemini-2.5-flash"
    client = create_client(load_api_key())
    chunk_embeddings = emb_model.encode(chunks)
    print()

    querys = ["What is the refund policy?", "How many vacation days do employees receive?",
              "When are performance reviews conducted?", "What are the office hours?",
                "How much training budget is available?", 
                "How much notice is required before resignation?", "exit"]
    for query in querys:
        if query == "exit":
            break
        result = retrieve(model=emb_model, query=query,chunks=chunks, chunk_embeddings=chunk_embeddings)
        if result is str():
            print(result)
        else:
            print(f"Query: {query}")
            print(f"Best Score: {result['score']}")
            print(f"Retrieved Chunk: {result['chunk']}")

            retrieve_prompt = build_prompt(query=query, context=result["chunk"])

            response = send_prompt(client=client, model=gen_model, prompt=retrieve_prompt)

            print(response.text)

if __name__ == "__main__":
    main()