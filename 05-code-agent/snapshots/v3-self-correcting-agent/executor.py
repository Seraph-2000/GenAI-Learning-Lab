import os
import uuid
import subprocess

class BaseExecutor:

    def execute(self, code: str):
        raise NotImplementedError
    
class PythonExecutor(BaseExecutor):

    def _create_temp_file(self, code):
        filename = f"tempfile_{str(uuid.uuid4())[:8]}.py"
        directory = os.getcwd()
        file_path = os.path.join(directory, filename)
        with open(file_path, "w") as f:
            f.write(code)

        return file_path

    def _run_file(self, filepath):
        output = subprocess.run(["python", filepath],
                                    capture_output=True, 
                                    text=True)
        return {
            "success":output.returncode == 0 ,
            "stdout": output.stdout.strip(),
            "stderr": output.stderr.strip(),
            "return_code": output.returncode
        }

    def _cleanup_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    def execute(self, code):
        try:
            file = None
            file = self._create_temp_file(code=code)

            result = self._run_file(filepath=file)
        finally:
            if file:
                self._cleanup_file(file_path=file)
            
        return result