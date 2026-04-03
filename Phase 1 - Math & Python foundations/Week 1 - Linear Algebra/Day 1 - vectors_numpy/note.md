# Vector

> What is a vector?
> A vector is an ordered list of numbers with a direction and a magnitude. In 2D space, the vector [3, 4] means "move 3 right, 4 up." In 768D space (a word embedding), it means "move to this precise position in meaning space."

## 3 Perspective of vector

- **Physics student perspective:** Vectors are arrows pointing in space, what defines a given vector it's length & direction

- **Computer science perspective:** Vectors are ordered list of numbers

- **Mathematician Perspective:** A vector is any object that follows linear algebra rules (addition and scaling). It is an abstract member of a "vector space."Mental Model: $v + w$, regardless of whether they are arrows, numbers, or functions.

## Scalar vs Vector vs Matrix vs Tensor

Scalar — a single number. e.g Temperature (37.2°C), Learning rate (0.001)
Vector — 1D array of numbers. One image row, one word embedding
Matrix — 2D grid. A grayscale image, a weight layer in a neural net

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

Tensor — ND generalization. An RGB image is a (3, H, W) tensor. A batch of images is (N, 3, H, W)

## The operations that power everything

- Dot product — measures similarity. Used in attention, cosine similarity, SVMs
- Matrix multiplication — every single linear layer in every neural net is a matmul
- Transpose — flips rows to columns. Critical for shape alignment
- Norm (magnitude) — length of a vector. Used in L2 regularization, normalization
