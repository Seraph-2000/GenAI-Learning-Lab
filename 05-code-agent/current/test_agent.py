import os
from dotenv import load_dotenv
from llm import GeminiLLM
from agent import CodeAgent

load_dotenv()

def load_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API Key not found! Please check your .env file.")
    return api_key

api_key = load_api_key()
model = "gemini-3.1-flash-lite"

llm = GeminiLLM(api_key=api_key, model=model)

agent = CodeAgent(llm)

response = agent.run("")
print(response)