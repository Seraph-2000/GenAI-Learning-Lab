import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def load_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API Key not found! Please check your .env file.")
    return api_key

def create_client(api_key):
    client = genai.Client(api_key = api_key)
    return client

def format_history(history, n = 10):
    formated_history = ""
    
    for message in history[-n:]:
        formated_history += message["role"] +" : " + message["content"] + "\n"

    return formated_history

def get_system_prompt():
    return """You are a teaching assistant."""
  
def calculate(expression):
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

def llm_route(client, model, prompt):

    router_prompt = f"""You are a routing assistant.

                    Available tools:
                    - calculator

                    Return ONLY one word:
                    calculator
                    none

                    User Input:
                    {prompt}"""
    return send_prompt(client, model, system_prompt=router_prompt)
    

TOOLS = {
    "calculator": calculate
    }


def execute_tool(tool_name, prompt):
    return TOOLS[tool_name](prompt)


def send_prompt(client, model, system_prompt, formated_history = " "):
    

    response = client.models.generate_content(
        model= model,
        contents= system_prompt + "\n\n" + formated_history
    )

    return response

print("Script started")

def main():
    history = []  
    model = "gemini-2.5-flash"
    client = create_client(load_api_key())
    relevancy = 10
    sys_prompt = get_system_prompt()

    while True:
        prompt = input("User : ") # 99*25

        if prompt == "exit":
            break
        
        history.append({"role" : "User", "content" : prompt})
        formated_history = format_history(history, relevancy)
        route_response =  llm_route(client=client, model=model, prompt=prompt)
        tool_name = (
            route_response.text
            .lower()
            .strip()
            .replace(".", "")
        )
        if tool_name == "calculator":
            response = execute_tool(tool_name=tool_name, prompt=prompt)
            history.append({"role" : "Assistant", "content" : str(response)})
            print(response)
        else:
            response = send_prompt(client, model, system_prompt=sys_prompt, formated_history=formated_history)
            history.append({"role" : "Assistant", "content" : response.text})
            print(response.text)

if __name__ == "__main__":
    main()