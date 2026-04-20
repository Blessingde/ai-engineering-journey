# Vector

> What is a vector?
> A vector is an ordered list of numbers with a direction and a magnitude. In 2D space, the vector [3, 4] means "move 3 right, 4 up." In 768D space (a word embedding), it means "move to this precise position in meaning space."

## 3 Perspective of vector

- **Physics student perspective:** Vectors are arrows pointing in space, what defines a given vector it's length & direction

- **Computer science perspective:** Vectors are ordered list of numbers

- **Mathematician Perspective:** A vector is any object that follows linear algebra rules (addition and scaling). It is an abstract member of a "vector space."Mental Model: $v + w$, regardless of whether they are arrows, numbers, or functions.

## Scalar vs Vector vs Matrix vs Tensor

### 1. What is a Scalar?

A single number. e.g Temperature (37.2°C), Learning rate (0.001)

### 2. What is a Vector?

1D array of numbers. One image row, one word embedding. A vector is just a list of numbers (usually in a single row or column).

### Examples:

$$
v =\begin{bmatrix}
a & b \\
\end{bmatrix}
$$

> This is a 1D vector

$$
v = \begin{bmatrix}
1 \\2\\3
\end{bmatrix}
$$

### This is a 3D vector

### Key Idea

A vector can have any number of dimensions, not just 2D or 3D.

> What “dimension” means here  
> Dimension = number of features

So:

> 2D vector → 2 features  
> 3D vector → 3 features  
> 10D vector → 10 features  
> 1000D vector → 1000 features

### Example (10D Vector)

$$
𝑥 =\begin{bmatrix}
𝑥1\\𝑥2\\𝑥3\\𝑥4\\𝑥5\\𝑥6\\𝑥7\\𝑥8\\𝑥9\\𝑥10
\end{bmatrix}
$$

#### This is a 10-dimensional vector

> Interpretation:

- Can represent:
- Position (x, y, z)<br>
- Features in ML
- Direction + magnitude

> **Final Understanding**  
> A vector in ML is just a compact way to store features of one data point
> <br> **One-Line Summary**  
> Each vector = one example, each number = one feature.

### 3. What is a Matrix?

2D grid. A grayscale image, a weight layer in a neural net. It is a rectangular table of numbers (rows × columns).

### Example:

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

This is a **2 × 2 matrix**

---

### 4. What is a Tensor?

It is a generalization of scalars, vectors, and matrices. An RGB image is a (3, H, W) tensor. A batch of images is (N, 3, H, W)

### Levels:

- Scalar → 0D
- Vector → 1D
- Matrix → 2D
- Tensor → 3D+

### Example:

- Image → (Height × Width × Channels)

Used in:

- Deep learning
- Computer vision
- Neural networks

---

## 🔹 5. Key Differences

| Feature   | Scalar       | Vector                            | Matrix         | Tensor            |
| --------- | ------------ | --------------------------------- | -------------- | ----------------- |
| Structure | Single value | 1 row/column                      | 2D table       | Multi-dimensional |
| Shape     | 1            | n×1 or 1×n                        | m×n            | nD                |
| Use       | Value        | Represent a points/direction/data | Transformation | Complex data      |

## The operations that power everything

- **Dot product:** measures similarity. Used in attention, cosine similarity, SVMs

- **Matrix multiplication:** every single linear layer in every neural net is a matmul

- **Transpose:** flips rows to columns. Critical for shape alignment

- **Norm (magnitude):** length of a vector. Used in L2 regularization, normalization
