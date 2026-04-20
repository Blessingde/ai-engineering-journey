import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(42)
database = np.random.randn(20, 5)
print(database)

labels = [f"label_{i}" for i in range(20)]

def cosine_similarity_matrix(query, database):
    q_norm = query / np.linalg.norm(query)
    d_norm = database / np.linalg.norm(database)

    similarities = d_norm @ q_norm
    return similarities


def search(query, database, labels, top_k=3):
    """Find the top_k most similar items to the query."""
    scores = cosine_similarity_matrix(query, database)

    top_indices = np.argsort(scores)[::-1][:top_k]
  
    print(f"Query: {query}")
    print(f"\nTop {top_k} results:")
    for rank, idx in enumerate(top_indices):
        print(f" {rank+1}., {labels[idx]} (similarity: {scores[idx]:.4f})")
    return top_indices, scores

#  Run a search
query_vector = np.random.randn(5)
search(query_vector, database, labels)

# Visualize similarity scores
all_scores = cosine_similarity_matrix(query_vector, database)
plt.figure(figsize=(10, 3))
colors = ['#534AB7' if s > 0.5 else '#B4B2A9' for s in all_scores]
plt.bar(labels, all_scores, color=colors)
plt.axhline(0, color='gray', linewidth=0.5)
plt.title('Cosine similarity: query vs. database')
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.savefig('day01_similarity_search.png', dpi=150)
plt.show()
print("Plot saved!")



