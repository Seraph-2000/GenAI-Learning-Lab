from llm import BaseLLM
from executor import BaseExecutor

class CodeAgent:

    def __init__(self, llm, executor, max_retries = 3):
        if not isinstance(llm, BaseLLM):
            raise TypeError("Expected BaseLLM implementation")
        self.llm = llm
        if not isinstance(executor, BaseExecutor):
            raise TypeError("Expected BaseExecutor implementation")
        self.executor = executor
        self.max_retries = max_retries
    

    def _build_prompt(self,task):
        prompt = f"""You are an expert Python developer.

            Write clean, executable Python code.

            Task:
            {task}

            Return only the code."""
        return prompt
    
    def _build_reflection_prompt(self, task, code, error):
        reflection = f"""
You are an expert Python debugging assistant.

The following code was generated to solve a task
but failed during execution.

Original Task:
{task}

Generated Code:
{code}

Execution Error:
{error}

Fix the code while preserving the original intent
of the task.

Return ONLY executable Python code.

Do not include explanations.
Do not include markdown.
Do not include code fences.
"""
        return reflection
    
    def run(self,task):
        if not task or not task.strip():
            return {
                "status": "error",
                "task": task,
                "message": "Task cannot be empty"
            }
        prompt = self._build_prompt(task=task)
        code = self.llm.generate(prompt=prompt)

        response = self.executor.execute(code=code)

        attempt_history = []
        attempt_history.append(
            {
                "attempt": 1,
                "generated_code": code,
                "execution_result": response
            }
        )
        reflect_prompt = None
        reflect_code = None
        reflect_response = None

        if response["success"]:
            return {
                "status": "success",
                "task": task,
                "prompt_used": prompt,
                "generated_code": code,
                "execution_result": response,
                "attempt_history": attempt_history
            }
        else:
            for i in range(2, self.max_retries + 2):
                reflect_prompt = self._build_reflection_prompt(task=task, code=attempt_history[-1]["generated_code"], error=attempt_history[-1]["execution_result"]["stderr"])
                reflect_code = self.llm.generate(prompt=reflect_prompt)
                reflect_response = self.executor.execute(code=reflect_code)
                attempt_history.append(
                    {
                        "attempt": i,
                        "generated_code": reflect_code,
                        "execution_result": reflect_response
                    }
                )
                if reflect_response["success"]:
                    return {
                        "status": "success",
                        "task": task,
                        "prompt_used": reflect_prompt,
                        "generated_code": reflect_code,
                        "execution_result": reflect_response,
                        "attempt_history": attempt_history
                    }
        return {
            "status": "failed",
            "task": task,
            "prompt_used": reflect_prompt,
            "generated_code": reflect_code,
            "execution_result": reflect_response,
            "attempt_history": attempt_history
            }