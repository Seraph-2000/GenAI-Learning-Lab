from executor import PythonExecutor

executor = PythonExecutor()

result = executor.execute("""
print(x)
""")

print(result)