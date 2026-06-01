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

def format_history(history):
    formated_history = ""
    
    for message in history:
        formated_history += message["role"] +" : " + message["content"] + "\n"

    return formated_history



def send_prompt(client, model, prompt):
    

    response = client.models.generate_content(
        model= model,
        contents=prompt
    )

    return response

print("Script started")

def main():
    history = []  
    model = "gemini-2.5-flash"
    client = create_client(load_api_key())
    
    while True:
        prompt = input("User : ")
        history.append({"role" : "User", "content" : prompt})
        print("DEBUG : HISTORY - ", history)
        if prompt == "exit":
            break
        formated_history = format_history(history)
        print("DEBUG : FORMATED HISTORY - ", formated_history)
        response = send_prompt(client, model, formated_history)
        history.append({"role" : "Assistant", "content" : response.text})
        
        print(response.text)

if __name__ == "__main__":
    main()