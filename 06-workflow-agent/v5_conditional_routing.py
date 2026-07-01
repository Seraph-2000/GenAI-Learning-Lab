from dataclasses import dataclass
from typing import Callable
from collections import defaultdict

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

def start(state):
    state.current_node = "START"
    state.execution_history.append("START")

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

def evaluate(state):
    state.current_node = "EVALUATOR"
    state.execution_history.append("EVALUATOR")
    if len(state.documents)>=2:
        return "GENERATOR"
    else:
        return "RETRIEVER"

def end(state):
    state.current_node = "END"
    state.execution_history.append("END")

class WorkflowGraph:
    def __init__(self):
        self.nodes = dict()
        self.edges = defaultdict(list)

    def add_node(self, node):
        self.nodes[node.name] = node

    def add_edge(self, source, destination):
        if self.edges[source.name]:
            if destination.name in self.edges[source.name]:
                print("connection exists")
            else:
                self.edges[source.name].append(destination.name)
        else:
            self.edges[source.name].append(destination.name)

    def get_node(self, name):
        return self.nodes[name]

    def get_next_nodes(self, node):
        if self.edges[node.name]:
            return self.edges[node.name]
        else:
            return []
    def get_start_node(self):
        return self.nodes["START"]
    
class WorkflowEngine:
    def __init__(self, graph):
        self.graph = graph
    
    def run(self, state):
        node = self.graph.get_start_node()
        while True:
            result = node.execute(state)
            if result is None:
                next_node = self.graph.get_next_nodes(node=node)
                if not next_node:
                    break
                else:
                    node = self.graph.get_node(next_node[0])
            else:
                if result in self.graph.edges[node.name]:
                    next_node = result
                    node = self.graph.get_node(next_node)
                else:
                    print("Error in routing")
                    break

state = WorkflowState(query = "What is RAG?", documents = [], answer = "",
                       current_node = "", execution_history = [])

start_node = Node(name = "START", function = start)
retriever_node = Node(name = "RETRIEVER", function = retrieve)
generator_node = Node(name = "GENERATOR", function = generate)
evaluator_node = Node(name = "EVALUATOR", function = evaluate)
end_node = Node(name = "END", function = end)

graph = WorkflowGraph()
nodes = [start_node, retriever_node, evaluator_node, generator_node, end_node]
for node in nodes:
    graph.add_node(node=node)

graph.add_edge(start_node, retriever_node)
graph.add_edge(retriever_node, evaluator_node)
graph.add_edge(evaluator_node, retriever_node)
graph.add_edge(evaluator_node, generator_node)
graph.add_edge(generator_node, end_node)


engine = WorkflowEngine(graph)
engine.run(state=state)

print(state)