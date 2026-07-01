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
TOP_K = 3
DOCUMENTS = [

    "Retrieval-Augmented Generation (RAG) combines information retrieval with large language models to generate grounded answers.",

    "A vector database stores embeddings and enables semantic similarity search.",

    "Embeddings are dense numerical representations that capture semantic meaning of text.",

    "BM25 is a lexical retrieval algorithm that ranks documents using keyword relevance.",

    "Hybrid search combines BM25 with vector search to improve retrieval accuracy.",

    "A reranker reorders retrieved documents based on semantic relevance before passing them to the LLM.",

    "A workflow graph represents an AI workflow as connected nodes and edges.",

    "Workflow state stores shared information that every node can read and update.",

    "Planner nodes decide which actions or tools should be executed next.",

    "Retriever nodes search external knowledge sources for relevant context.",

    "Evaluator nodes determine whether enough information has been collected to continue.",

    "Generator nodes use retrieved context to produce a final response.",

    "Reflection nodes critique generated responses and determine whether another iteration is required.",

    "Conditional routing allows workflows to branch dynamically based on the current workflow state.",

    "Checkpointing enables long-running workflows to resume from previously saved state.",

    "Human-in-the-loop systems pause execution until a human approves or modifies the workflow.",

    "LangGraph models AI applications as stateful workflow graphs instead of simple sequential chains.",

    "Agent orchestration coordinates planning, retrieval, reasoning, and generation across multiple workflow steps.",

    "Shared state allows independent workflow nodes to exchange information without directly calling one another.",

    "Production AI systems often include retries, evaluation loops, observability, and human approval steps."

]

def start(state):
    state.current_node = "START"
    state.execution_history.append("START")

def planner(state):
    state.current_node = "PLANNER"
    state.execution_history.append("PLANNER")
    if "what" in state.query.lower():
        return "RETRIEVER"
    else:
        return "GENERATOR"

def retrieve(state):
    """
    Simulates document retrieval.

    Production:
        - Embed query
        - Search vector database
        - Return top-k documents
    """
    retrieved_documents = []
    query = state.query.lower().split()
    for doc in DOCUMENTS:
        score = 0
        for key_word in query:
            if key_word in doc.lower():
                score += 1
        if score > 0:
            retrieved_documents.append((score, doc))
        
    retrieved_documents = sorted(retrieved_documents, reverse=True)
    final_retrieved_documents = [item[1] for item in retrieved_documents[:TOP_K]] 

    state.current_node = "RETRIEVER"
    state.documents = final_retrieved_documents
    state.execution_history.append("RETRIEVER")

def generate(state):
    """
    Simulates answer generation.

    Production:
        - Send query + retrieved context to an LLM
        - Store generated response in state
    """
    state.current_node = "GENERATOR"
    state.answer = (
        "Answer generated using "
        + str(len(state.documents))
        + " retrieved documents."
    )
    state.execution_history.append("GENERATOR")

def evaluate(state):
    state.current_node = "EVALUATOR"
    state.execution_history.append("EVALUATOR")
    if len(state.documents)>=2:
        return "GENERATOR"
    else:
        return "RETRIEVER"
    
def reflect(state):
    state.current_node = "REFLECT"
    state.execution_history.append("REFLECT")
    if len(state.answer) > 30:
        return "END"
    else:
        return "GENERATOR"

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
planner_node = Node(name = "PLANNER", function = planner)
retriever_node = Node(name = "RETRIEVER", function = retrieve)
generator_node = Node(name = "GENERATOR", function = generate)
evaluator_node = Node(name = "EVALUATOR", function = evaluate)
reflect_node = Node(name = "REFLECT", function = reflect)
end_node = Node(name = "END", function = end)

graph = WorkflowGraph()
nodes = [start_node, planner_node, retriever_node, evaluator_node,
          generator_node, reflect_node, end_node]
for node in nodes:
    graph.add_node(node=node)

graph.add_edge(start_node, planner_node)
graph.add_edge(planner_node, retriever_node)
graph.add_edge(planner_node, generator_node)
graph.add_edge(retriever_node, evaluator_node)
graph.add_edge(evaluator_node, retriever_node)
graph.add_edge(evaluator_node, generator_node)
graph.add_edge(generator_node, reflect_node)
graph.add_edge(reflect_node, generator_node)
graph.add_edge(reflect_node, end_node)


engine = WorkflowEngine(graph)
engine.run(state=state)

print(state)