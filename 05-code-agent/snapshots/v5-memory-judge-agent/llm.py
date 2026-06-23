from google import genai

class BaseLLM:
    
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class GeminiLLM(BaseLLM):

    def __init__(self, api_key, model):
        self.model = model
        self.client = genai.Client(api_key=api_key)
    
    def generate(self, prompt: str) -> str:

        response = self.client.models.generate_content(model=self.model, contents=prompt)

        return response.text