"""
Task 1: Without running code first, manually calculate the dot product of [2, 3, 5] and [1, 4, 2]. 
Then verify with NumPy. If wrong, trace where your calculation diverged.
"""
a = np.array([2, 3, 5])
b = np.array([1, 4, 2])

print(f'dot_product of vector a and b: {a@b}')

"""
Task 2: Create a (5 × 3) random matrix W and a (3,) input vector x. Compute the forward pass W @ x. 
Then create a batch of 10 inputs (shape 10×3) and compute all outputs at once. Verify the output shapes.
"""
np.random.seed(10)
W = np.random.randn(5, 3) # Weight matrix: 5 output neurons and 3 input features
x = np.array([2, 5, 4]) #
output = W @ x
print(f"Outshape for one input: {output.shape}")

X_batch = np.random.randn(10,3)
outputs = X_batch @ W.T # or W @ X_batch.T
print(f"Outshape for ten input: {outputs.shape}")

"""
Task 3: Implement cosine_similarity from scratch using only basic NumPy (np.dot, np.linalg.norm). 
Then use it to find which of these 3 vectors is most similar to [1, 0, 0]: [0.9, 0.1, 0], [0, 1, 0], 
[0.5, 0.5, 0.5].
"""
def cosine_similarity(a, my_tuple):
  sim_scores = []
  for i in my_tuple:
        sim_score = np.dot(a, i) / (np.linalg.norm(a) * np.linalg.norm(i))
        sim_scores.append(sim_score)
  return max(sim_scores), sim_scores

a = [1, 0, 0]
my_tuple = ([0.9, 0.1, 0], [0, 1, 0], [0.5, 0.5, 0.5])
print(cosine_similarity(a, my_tuple)[0])
print(cosine_similarity(a, my_tuple)[1])

"""
Task 4: Write a function normalize_batch(X) that takes a (N × D) matrix and returns it 
normalized so each feature (column) has mean ≈ 0 and std ≈ 1. This is z-score normalization — the 
first operation in almost every ML pipeline.
"""
import numpy as np

def normalize_batch(X):
    # Calculate mean and std for each column (axis=0)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
 
    X_normalized = (X - mu) / sigma 
    return X_normalized
"""
Task 5 (challenge): Without using np.matmul or @, implement matrix multiplication manually 
using only nested for loops and elementwise operations. Then verify it matches np.matmul on a 3×3 
example. This is how you know you really understand it.
"""