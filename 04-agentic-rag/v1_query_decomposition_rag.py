import os
import json
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

with open("documents.json", "r") as f:
    documents = json.load(f) 

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

def retrieve(model, query, documents, document_embeddings):
    embeded_query = model.encode(query)
    threshold = 0.3
    top_k = 3
    results = []
    for i in range(len(documents)):
        similarity = cosine_similarity([embeded_query], [document_embeddings[i]]).item()
        if similarity < threshold:
            continue
        else:
            results.append({
                "chunk": documents[i]["content"],
                "score": similarity,
                "source": documents[i]["source"]
                })
    if results == []:
        return "No relevent document found"
    else:
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
def decompose_query(client, model, query):
    decompose_prompt = f"""
        You are a query decomposition system.

        Break the user's question into retrieval-focused subqueries.

        Return ONLY valid JSON.

        Example:

        User:
        Compare refund policy and vacation policy.

        Output:
        [
            {{
                "query":"refund policy",
                "reason":"Need refund information"
            }},
            {{
                "query":"vacation policy",
                "reason":"Need vacation information"
            }}
        ]

        User:
        {query}
        """
    response = send_prompt(client=client, model=model, prompt=decompose_prompt)
    return json.loads(response.text)

    
def main():
    print("started script")

    
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_model = "gemini-3.1-flash-lite" #"gemini-2.5-flash"
    client = create_client(load_api_key())
    document_embeddings = emb_model.encode([doc["content"] for doc in documents])
    print()
    history = []
    while True:
        user_query = input("User : ")

        if user_query == "exit":
            break

        history.append({"role" : "User", "content" : user_query})


        subqueries = decompose_query(query=user_query, client=client, model=gen_model)
        
        all_res = []
        all_sources = set()
        for subquery in subqueries:
            print("\nSubquery:")
            print(subquery["query"])

            print("Reason:")
            print(subquery["reason"])


            results = retrieve(model=emb_model, query=subquery["query"], documents=documents, document_embeddings=document_embeddings)
            
            if isinstance(results,str):
                all_res.append({
                    "query": subquery["query"],
                    "reason": subquery["reason"],
                    "result": results
                })
            
            else:
                context = "\n\n".join(result["chunk"] for result in results)

                memory = format_history(history=history)

                retrieve_prompt = build_prompt(query=subquery["query"], context=context, memory=memory)

                response = send_prompt(client=client, model=gen_model, prompt=retrieve_prompt)

                all_res.append({
                    "query": subquery["query"],
                    "reason": subquery["reason"],
                    "result": response.text
                })
                

                # Get unique sources
                all_sources.update(result["source"] for result in results)

        formatted_results = ""

        for result in all_res:
            formatted_results += (
                f"Query: {result['query']}\n"
                f"Reason: {result['reason']}\n"
                f"Result: {result['result']}\n\n"
            )
        synthesis_prompt = f"""
            User Question:
            {user_query}

            Retrival Results:
            {formatted_results}

            Answer the user's question using the retrival results.
            Combine information if necessary.
            """
        response = send_prompt(
                    client=client,
                    model=gen_model,
                    prompt=synthesis_prompt
                )
        history.append({
            "role": "Assistant",
            "content": response.text
        })

        print(response.text)    

        print("\nSources:")
        for source in all_sources:
            print(f"- {source}")

if __name__ == "__main__":
    main()