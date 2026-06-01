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

def send_prompt(client, model, prompt):
    

    response = client.models.generate_content(
        model= model,
        contents=prompt
    )

    return response

print("Script started")

def main():
    prompt = "Explain how AI works in a few words"
    model = "gemini-2.5-flash"
    client = create_client(load_api_key())

    response = send_prompt(client, model, prompt)
    
    
    print(response.text)

if __name__ == "__main__":
    main()