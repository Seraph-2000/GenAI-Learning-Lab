from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

document = """
Refund Policy
Customers can request refunds within 30 days.

Work From Home Policy
Employees may work remotely three days per week.

Vacation Policy
Employees receive 20 days of paid leave.
"""

def retrieve(model, query, chunks, chunk_embeddings):
    embeded_query = model.encode(query)
    best_score = 0
    best_index = 0
    for i in range(len(chunks)):
        similarity = cosine_similarity([embeded_query], [chunk_embeddings[i]])
        if best_score<similarity:
            best_score = similarity
            best_index = i
    return chunks[best_index]


def main():
    print("started script")
    chunks = document.split("\n\n")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("\n\nstarted embedding chunks\n\n")
    chunk_embeddings = model.encode(chunks)

    query = "What is the refund policy?"

    response = retrieve(model=model, query=query,chunks=chunks, chunk_embeddings=chunk_embeddings)
    
    print(response)

if __name__ == "__main__":
    main()