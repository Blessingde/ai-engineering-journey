import numpy as np
import matplotlib.pyplot as plt

# Scalar, vectors, matrix and tensors
scaler = 42
vector =  np.array([1,2, 4, 5])
matrix = np.array([[1,2,3],
                    [4,5,6],
                    [7,8,9]])
tensor = np.zeros((3, 64, 64))

print(vector.shape)   # (5,)
print(matrix.shape)   # (3, 3)
print(tensor.shape)   # (3, 64, 64)
print(matrix.dtype)   # float64 or int64

# Useful constructors
zeros    = np.zeros((3, 3))
ones     = np.ones((2, 4))
identity = np.eye(3)              # identity matrix
randn    = np.random.randn(3, 3)  # standard normal — used to init weights
arange   = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]

# Vector operation
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# Element-wise operation
print(f'Element-wise Addition: {a+b}')
print(f'Element-wise Multiplication: {a+b}')
print(f'a-squared: {a ** 2}')

# Dot product (similarity measure)
dot = np.dot(a, b)
dot_alt = a @ b # same thing, @ is the matmul (matrix multiplication) operator
print(f"Dot product: {dot}")

# Norm and normalization
norm_a = np.linalg.norm(a)          # 3.7416...
unit_a = a / norm_a                  # unit vector
print(f"Norm: {norm_a:.4f}")
print(f"Unit vector: {unit_a}")
print(f"Unit norm check: {np.linalg.norm(unit_a):.4f}")  # should be 1.0

# Cosine similarity (used in NLP, search, embeddings)
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"Cosine similarity: {cosine_similarity(a, b):.4f}")
print(f"Similarity with itself: {cosine_similarity(a, a):.4f}")  # 1.0

# Matrix operation
W = np.random.randn(4, 3)   # weight matrix: 4 output neurons, 3 input features
x = np.array([1.0, 2.0, 3.0])  # one input sample (3 features)

# Linear layer forward pass — this is what nn.Linear does internally
output = W @ x
print(f"Input shape: {x.shape}")     # (3,)
print(f"Weight shape: {W.shape}")    # (4, 3)
print(f"Output shape: {output.shape}")  # (4,) — 4 neuron activations

# Batch of inputs (multiple samples at once)
X_batch = np.random.randn(8, 3)   # 8 samples, 3 features each
outputs  = X_batch @ W.T           # (8,3) @ (3,4) = (8,4)
print(f"Batch output: {outputs.shape}")  # (8, 4)

# Matrix inverse and solving linear systems
A = np.array([[2.0, 1.0], [1.0, 3.0]])
b_vec = np.array([5.0, 10.0])
x_sol = np.linalg.solve(A, b_vec)  # solve Ax = b
print(f"Solution: {x_sol}")

# Broadcasting: NumPy stretches smaller arrays to match larger ones
# This is how batch normalization, bias addition, etc. work

matrix = np.ones((3, 4))
bias   = np.array([1.0, 2.0, 3.0, 4.0])  # shape (4,)

result = matrix + bias   # (3,4) + (4,) → bias added to every row
print(result)

# Subtracting the mean (used in normalization constantly)
data = np.random.randn(100, 5)   # 100 samples, 5 features
mean = data.mean(axis=0)         # mean per feature, shape (5,)
std  = data.std(axis=0)          # std per feature, shape (5,)
normalized = (data - mean) / std  # broadcast across 100 rows
print(f"Normalized mean: {normalized.mean(axis=0).round(6)}")  # ≈ 0
print(f"Normalized std:  {normalized.std(axis=0).round(6)}")   # ≈ 1



