from llm import BaseLLM
from executor import BaseExecutor
import json


class CodeAgent:

    def __init__(self, llm, executor, max_retries=3):
        if not isinstance(llm, BaseLLM):
            raise TypeError("Expected BaseLLM implementation")

        if not isinstance(executor, BaseExecutor):
            raise TypeError("Expected BaseExecutor implementation")

        self.llm = llm
        self.executor = executor
        self.max_retries = max_retries

    def _build_prompt(self, task):
        prompt = f"""
You are an expert Python developer.

Write clean, executable Python code.

Task:
{task}

Return only the code."""
        return prompt

    def _record_attempt(self, attempt_history, attempt_number, code, execution_result, judge_result=None):
        attempt = {
            "attempt": attempt_number,
            "generated_code": code,
            "execution_result": execution_result
        }

        if judge_result is not None:
            attempt["judge_result"] = judge_result

        attempt_history.append(attempt)

    def _get_recent_memory(self, memory, n):
        return memory[-n:]

    def _build_reflection_prompt(self, task, code, error, memory, feedback):
        memory_text = "Recent Failed Attempts:\n"

        for m in memory:
            memory_text += f"""

Attempt {m["attempt"]}

Code:
{m["generated_code"]}

Execution Error:
{m["execution_result"].get("stderr", "")}
"""

            if "judge_result" in m:
                memory_text += f"""

Judge Feedback:
{m["judge_result"]["reason"]}
"""

        judge_feedback = ""

        if feedback is not None:
            judge_feedback = f"""

Latest Judge Feedback:
{feedback["reason"]}
"""

        reflection = f"""
You are an expert Python debugging assistant.

The following code failed to satisfy the task requirements.

{memory_text}

Original Task:
{task}

Generated Code:
{code}

Execution Error:
{error}

{judge_feedback}

Fix the code while preserving the original intent of the task.

Return ONLY executable Python code.

Do not include explanations.
Do not include markdown.
Do not include code fences.
"""

        return reflection

    def _judge_result(self, task, code, result):

        judge_prompt = f"""
You are an expert code evaluator.

Original Task:
{task}

Generated Code:
{code}

Execution Result:
{result}

Determine whether the code successfully fulfilled the task.

Return ONLY valid JSON.

Example Success:

{{
    "status": "passed",
    "reason": "The code satisfies the task."
}}

Example Failure:

{{
    "status": "failed",
    "reason": "The code executed but did not satisfy the task."
}}
"""

        response = self.llm.generate(prompt=judge_prompt)

        try:
            judge_result = json.loads(response)

        except json.JSONDecodeError:
            return {
                "status": "judge_error",
                "reason": "Judge output could not be parsed."
            }

        return judge_result

    def run(self, task):

        if not task or not task.strip():
            return {
                "status": "error",
                "task": task,
                "message": "Task cannot be empty"
            }

        prompt = self._build_prompt(task=task)

        code = self.llm.generate(prompt=prompt)

        response = self.executor.execute(code=code)

        judge_result = self._judge_result(
            task=task,
            code=code,
            result=response
        )

        if judge_result["status"] == "judge_error":
            return judge_result

        attempt_history = []

        self._record_attempt(
            attempt_history=attempt_history,
            attempt_number=1,
            code=code,
            execution_result=response,
            judge_result=judge_result
        )

        if judge_result["status"] == "passed":
            return {
                "status": "success",
                "task": task,
                "prompt_used": prompt,
                "generated_code": code,
                "execution_result": response,
                "attempt_history": attempt_history
            }

        reflect_prompt = None
        reflect_code = None
        reflect_response = None
        reflect_results = None

        for i in range(2, self.max_retries + 2):

            reflect_prompt = self._build_reflection_prompt(
                task=task,
                code=attempt_history[-1]["generated_code"],
                error=attempt_history[-1]["execution_result"].get("stderr", ""),
                memory=self._get_recent_memory(attempt_history, 3),
                feedback=reflect_results if reflect_results else judge_result
            )

            reflect_code = self.llm.generate(prompt=reflect_prompt)

            reflect_response = self.executor.execute(code=reflect_code)

            reflect_results = self._judge_result(
                task=task,
                code=reflect_code,
                result=reflect_response
            )

            if reflect_results["status"] == "judge_error":
                return reflect_results

            self._record_attempt(
                attempt_history=attempt_history,
                attempt_number=i,
                code=reflect_code,
                execution_result=reflect_response,
                judge_result=reflect_results
            )

            if reflect_results["status"] == "passed":
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