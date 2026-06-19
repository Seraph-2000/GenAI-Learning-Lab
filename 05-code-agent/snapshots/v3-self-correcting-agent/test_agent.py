import os
from dotenv import load_dotenv
from llm import GeminiLLM
from agent import CodeAgent
from executor import PythonExecutor

executor = PythonExecutor()

load_dotenv()

def load_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API Key not found! Please check your .env file.")
    return api_key

api_key = load_api_key()
model = "gemini-3.1-flash-lite"

llm = GeminiLLM(api_key=api_key, model=model)


agent = CodeAgent(llm=llm, executor=executor)

response = agent.run("write python code that prints an undefined variable x")
print(response)