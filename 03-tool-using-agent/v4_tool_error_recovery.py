import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

load_dotenv()

documents = [
    {
        "source": "Refund Policy",
        "content": "Customers can request refunds within 30 days of purchase. Refunds are processed within 5 business days after approval."
    },
    {
        "source": "Work From Home Policy",
        "content": "Employees may work remotely up to three days per week with manager approval."
    },
    {
        "source": "Vacation Policy",
        "content": "Employees receive 20 days of paid leave annually. Unused leave can be carried forward for up to 5 days."
    },
    {
        "source": "Health Insurance Policy",
        "content": "The company provides medical insurance coverage for employees and their dependents."
    },
    {
        "source": "Travel Reimbursement Policy",
        "content": "Business travel expenses including flights, hotels, and meals will be reimbursed upon submission of receipts."
    },
    {
        "source": "Laptop Policy",
        "content": "Every employee is assigned a company laptop. Employees are responsible for maintaining and securing company devices."
    },
    {
        "source": "Training Policy",
        "content": "Employees are entitled to an annual training budget of $1000 for professional development courses and certifications."
    },
    {
        "source": "Performance Review Policy",
        "content": "Performance reviews are conducted twice a year, in June and December."
    },
    {
        "source": "Dress Code Policy",
        "content": "Employees are expected to maintain business casual attire while working from the office."
    },
    {
        "source": "Office Timing Policy",
        "content": "Regular office hours are from 9 AM to 6 PM, Monday through Friday."
    },
    {
        "source": "Resignation Policy",
        "content": "Employees must provide a minimum notice period of 30 days before leaving the company."
    },
    {
        "source": "Parental Leave Policy",
        "content": "Employees are entitled to 12 weeks of paid parental leave following the birth or adoption of a child."
    }
]

def load_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API Key not found! Please check your .env file.")
    return api_key

def create_client(api_key):
    client = genai.Client(api_key=api_key)
    return client

def send_prompt(client, model, prompt):
    response = client.models.generate_content(
        model= model,
        contents= prompt
        )
    return response

def calculator(expression):
    operators = ["+", "-", "*", "/"]
    try:
        for op in operators:
            if op in expression:

                left, right = expression.split(op)

                left = float(left.strip())
                right = float(right.strip())

                if op == "+":
                    return left + right

                elif op == "-":
                    return left - right

                elif op == "*":
                    return left * right

                elif op == "/":
                    return left / right
    except Exception:
        return "Invalid expression"

def search_docs(model, query, documents, document_embeddings):
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

TOOLS = {
    "calculator": calculator,
    "search_docs": search_docs
}

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

def llm_route(client, model, input):

    router_prompt = f"""
        You are a routing assistant.

        Available tools:
        - calculator
        - search_docs

        Return ONLY a valid JSON array.

        Examples:

        User Input:
        What is 25 * 89?

        Output:
        [
        {{
            "tool":"calculator",
            "query":"25 * 89"
        }}
        ]

        User Input:
        What is 25 * 89 and what is the refund policy?

        Output:
        [
        {{
            "tool":"calculator",
            "query":"25 * 89"
        }},
        {{
            "tool":"search_docs",
            "query":"refund policy"
        }}
        ]

        User Input:
        {input}
        """
    return send_prompt(client, model, prompt=router_prompt)

def repair_expression(client, model, query):
    repair_prompt = """You are an expression extraction system.

    Extract ONLY a valid arithmetic expression.

    Examples:

    Calculate 25 * 89 for me
    Output:
    25 * 89

    What is 100 / 4?
    Output:
    100 / 4

    If no valid arithmetic expression exists, return:
    Invalid Expression
    
    Entered Expression : """ + query
    response = send_prompt(client=client, model=model, prompt=repair_prompt)

    return response.text

def execute_tool(tool_call, model, documents, document_embeddings):

    tool_name = tool_call["tool"]

    if tool_name == "calculator":
        return calculator(tool_call["query"])

    elif tool_name == "search_docs":
        return search_docs(
            model=model,
            query=tool_call["query"],
            documents=documents,
            document_embeddings=document_embeddings
        )

def format_history(history):
    formated_history = ""
    
    for message in history:
        formated_history += message["role"] +" : " + message["content"] + "\n"

    return formated_history

def main():
    print("started script")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_model = "gemini-3.1-flash-lite" #"gemini-2.5-flash"
    client = create_client(load_api_key())
    document_embeddings = emb_model.encode([doc["content"] for doc in documents])
    
    print()
    history = []
    
    while True:
        query = input("User : ")

        if query == "exit":
            break

        history.append({"role" : "User", "content" : query})
        
        route_response = llm_route(client=client, model=gen_model, input=query)
        tool_calls = json.loads(route_response.text)

        all_results = []

        for tool_call in tool_calls:
            results = execute_tool(tool_call=tool_call, model=emb_model, documents=documents, document_embeddings=document_embeddings)

            print("\nTool Used:")   
            print(tool_call["tool"])
            
            if tool_call["tool"] == "calculator":

                if results == "Invalid expression":

                    repaired_query = repair_expression(client=client, model=gen_model, query=tool_call["query"])

                    if repaired_query.lower() != "invalid expression":

                        results = calculator(repaired_query)

                all_results.append({
                    "tool": tool_call["tool"],
                    "result": results
                })
            
            elif tool_call["tool"] == "search_docs":
                if isinstance(results, str):
                    all_results.append({
                        "tool": "search_docs",
                        "result": results
                    })

                    continue
                context = "\n\n".join(result["chunk"] for result in results)

                memory = format_history(history=history)

                retrieve_prompt = build_prompt(query=query, context=context, memory=memory)

                response = send_prompt(client=client, model=gen_model, prompt=retrieve_prompt)

                all_results.append({
                    "tool": tool_call["tool"],
                    "result": response.text
                })

                if response.text == "I don't know.":
                    continue
                else:
                    # Get unique sources
                    sources = list(set(result["source"] for result in results))

                    print("\nSources:")
                    for source in sources:
                        print(f"- {source}")

            elif tool_call["tool"] == "none":
                response = send_prompt(
                    client=client,
                    model=gen_model,
                    prompt=query
                    )
                all_results.append({
                    "tool": tool_call["tool"],
                    "result": response.text
                })
        formatted_results = ""

        for result in all_results:

            formatted_results += (
                f"Tool: {result['tool']}\n"
                f"Result: {result['result']}\n\n"
            )
        synthesis_prompt = f"""
            User Question:
            {query}

            Tool Results:
            {formatted_results}

            Answer the user's question using the tool results.
            Combine information from multiple tools if necessary.
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

if __name__ == "__main__":
    main()