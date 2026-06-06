from llm import BaseLLM

class CodeAgent:

    def __init__(self, llm):
        if not isinstance(llm, BaseLLM):
            raise TypeError("Expected BaseLLM implementation")
        self.llm = llm
    

    def _build_prompt(self,task):
        prompt = f"""You are an expert Python developer.

            Write clean, executable Python code.

            Task:
            {task}

            Return only the code."""
        return prompt
    
    def run(self,task):
        if not task or not task.strip():
            return {
                "status": "error",
                "task": task,
                "message": "Task cannot be empty"
            }
        prompt = self._build_prompt(task=task)
        response = self.llm.generate(prompt=prompt)
        return {
            "status": "success",
            "task": task,
            "prompt_used": prompt,
            "generated_code": response
        }