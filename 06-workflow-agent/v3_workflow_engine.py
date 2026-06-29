from dataclasses import dataclass
from typing import Callable

@dataclass
class WorkflowState:
    query : str
    documents : list[str]
    answer : str
    current_node : str
    execution_history : list[str]

@dataclass
class Node:
    name : str
    function : Callable

    def execute(self, state: WorkflowState):
        return self.function(state)

DOCUMENTS = [
    "RAG retrieves information from external sources",
    "Retrieved documents are provided to an LLM"
]

def retrieve(state):
    """
    Simulates document retrieval.

    Production:
        - Embed query
        - Search vector database
        - Return top-k documents
    """
    state.current_node = "RETRIEVER"
    state.documents.extend(DOCUMENTS)
    state.execution_history.append("RETRIEVER")

def generate(state):
    """
    Simulates answer generation.

    Production:
        - Send query + retrieved context to an LLM
        - Store generated response in state
    """
    state.current_node = "GENERATOR"
    state.answer = "RAG combines retrieval and generation."
    state.execution_history.append("GENERATOR")

class WorkflowEngine:
    
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
    
    def run(self, state):
        for node in self.nodes:
            node.execute(state)

state = WorkflowState(query = "What is RAG?", documents = [], answer = "",
                       current_node = "START", execution_history = ["START"])

retriever_node = Node(name = "RETRIEVER", function = retrieve)
generator_node = Node(name = "GENERATOR", function = generate)

nodes = [retriever_node, generator_node]

engine = WorkflowEngine(nodes=nodes)
engine.run(state=state)

print(state)