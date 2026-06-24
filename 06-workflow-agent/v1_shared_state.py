from dataclasses import dataclass
#from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class WorkflowState:
    query : str
    documents : list[str]
    answer : str
    current_node : str
    execution_history : list[str]

DOCUMENTS = [
    "RAG retrieves information from external sources",
    "Retrieved documents are provided to an LLM"
]

def retrieve(state):
    state.current_node = "RETRIEVER"
    state.documents.extend(DOCUMENTS)
    state.execution_history.append("RETRIEVER")

def generate(state):
    state.current_node = "GENERATOR"
    state.answer = "RAG combines retrieval and generation."
    state.execution_history.append("GENERATOR")

state = WorkflowState(query = "What is RAG?", documents = [], answer = "",
                       current_node = "START", execution_history = ["START"])

retrieve(state)
generate(state)
print(state)