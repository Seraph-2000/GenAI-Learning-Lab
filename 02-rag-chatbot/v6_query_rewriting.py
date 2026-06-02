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

def build_prompt(query, context, memory):
    return f"""
        You are a helpful assistant.

        Answer ONLY using the provided context and memory.

        If the answer is not present in the context or memory,
        say "I don't know."

        Context:
        {context}

        Memory:
        {memory}

        Question:
        {query}"""

def rewrite_query(client, model, prompt, memory):
    rewrite_prompt = f"""
    You are a query rewriting system.

    Your task:
    - Convert the current question into a standalone question.
    - Use the conversation history only if needed.
    
    Rules:
    - Preserve the user's original meaning.
    - Do not add new facts, assumptions, or details.
    - Do not infer quantities, dates, durations, or policies that were not explicitly mentioned.
    - Only replace references such as "it", "they", "that", "those", or similar ambiguous terms.
    - If the question is already understandable on its own, return it unchanged.
    - Return only the rewritten question.

    Conversation History:
    {memory}

    Current Question:
    {prompt}

    Standalone Question:
    """
    response = client.models.generate_content(
        model= model,
        contents= rewrite_prompt
        )
    print("DEBUG - ORIGINAL:", prompt)
    print("DEBUG - REWRITTEN:", response.text)
    
    return response.text

def send_prompt(client, model, prompt):
    response = client.models.generate_content(
        model= model,
        contents= prompt
        )

    return response

def format_history(history):
    formated_history = ""
    
    for message in history:
        formated_history += message["role"] +" : " + message["content"] + "\n"

    return formated_history

def retrieve(model, query, chunks, chunk_embeddings):
    embeded_query = model.encode(query)
    threshold = 0.3
    top_k = 3
    results = []
    for i in range(len(chunks)):
        similarity = cosine_similarity([embeded_query], [chunk_embeddings[i]]).item()
        if similarity < threshold:
            continue
        else:
            results.append({"chunk" : chunks[i],
                            "score" : similarity})
    if results == []:
        return "No relevent document found"
    else:
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

def main():
    print("started script")
    chunks = document.split("\n\n")
    
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_model = "gemini-3.1-flash-lite" #"gemini-2.5-flash"
    client = create_client(load_api_key())
    chunk_embeddings = emb_model.encode(chunks)
    print()
    history = []
    while True:
        query = input("User : ")

        if query == "exit":
            break

        history.append({"role" : "User", "content" : query})

        user_messages = []

        for message in history[:-1]:
            if message["role"] == "User":
                user_messages.append(message)

        query = rewrite_query(client=client, model=gen_model, prompt=query, memory=user_messages)
        
        
        results = retrieve(model=emb_model, query=query, chunks=chunks, chunk_embeddings=chunk_embeddings)
        
        if isinstance(results,str):
            history.append({"role" : "Assistant", "content" : results})
            print(results)
        
        else:
            context = "\n\n".join(result["chunk"] for result in results)

            memory = format_history(history=history)

            retrieve_prompt = build_prompt(query=query, context=context, memory=memory)

            response = send_prompt(client=client, model=gen_model, prompt=retrieve_prompt)

            history.append({"role" : "Assistant", "content" : response.text})

            print("Assistant : " + response.text)

if __name__ == "__main__":
    main()