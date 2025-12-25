## Lecture 01

### Defining Machine Learning and Artificial Intelligence

**Machine Learning (ML)** is the study of computer algorithms that **improve automatically through experience**, where experience is typically defined as prior training data. The term was coined by **Arthur Samuel** in 1959, who demonstrated its potential through a checkers program that learned to play better than its creator by playing against itself.

    - **Artificial Intelligence (AI) :**  A broad field focused on building programs that exhibit human-level cognitive performance. ML is a **subset** of AI that specifically uses data to achieve this performance. 
    -  **Deep Learning:** A further specialized sub-field of ML that utilizes **neural networks** and has driven significant recent progress in computer vision, speech recognition and language translation. 
----

### Taxonomy of Learning

Machine Learning problems are generally categorized into three main types based on the nature of the training signal:

    - **Supervised Learning:** The algorithm is provided with "labeled" training data consisting of **input-output pairs**. 

        - **Regression:** Predicting a **continuous, real-valued** output, such as wind speed or house prices.

        - **Classification:** Predicting a **discrete** class or category, such as weather it will rain (binary) or identifying handwritten digits in the **MNIST** dataset (multi-class).

    - **Unsupervised Learning:** The algorithm receives data without explicit labels and must find **patterns or hidden structures**, such as clusters or sub-spaces.

        - **The Cocktail Party Problem:** An example where an algorithm (like **ICA**) separates mixed audio signals from multiple microphones into distinct voices without knowing what the individual voices sound like beforehand. 

    - **Reinforcement Learning:** An agent learns to perform a task by interacting with an environment or simulator, adjusting its actions to maximize a reward, such as learning balance an **inverted  pendulum (cart-pole)**.

----

### Linear Algebra: Notation and Basic Operations

Linear Algebra serves as the language for representing data (e.g., the **design matrix** X, where rows are examples and columns are features) and performing optimizations.

- **Vectors (**$x \in \mathbb{R}^n$**):** By convention, these are treated as **column vectors**. A row vector is denoted by the transpose, $\mathbb{x}^T$.

- **Matrices (**$A \in \mathbb{R}^{mxn}$): A grid of real numbers that can be viewed as a collection of column vectors or row vectors.

- **Inner Product (Dot Product):** The product $x^T y$ results in a **scalar**.

- **Outer Product:** The product $xy^T$ results in a **rank-one matrix** where each entry $(i,j)$ is $x_i y_j$.

- **Special Matrices:** Includes the **Identity matrix** (I) with ones on the diagonal, **Diagonal matrices** (D), and **Symmetric matrices** where $A = A^T$.

----

### Interpretations of Matrix Multiplication

There are multiple ways to interpret $C = AB$;

- **Row Interpretation:** Each entry $C_{ij}$ is the inner product of the $i$-th row of $A$ and the $j$-th column of $B$.

- **Column Interpretation:** The columns of $C$ are **linear combinations** of the columns of $A$, weighted by the entries of $B$.

- **Outer Product Interpretation:** $AB$ is the **sum of the outer products** of $A$ and the rows of $B$.

----

### Geometric Intuition of Matrices and Subspaces

A matrix $A \in \mathbb{R}^{mxn}$ should be viewed as as **function** that maps a vector from an $n$-dimensional input space to an $m$-dimensional output space.

- **Full Rank:** If a square matrix is full rank, it creates a **one-to-one (bijective) mapping** between the input space and output spaces, making it **invertible** ($A^{-1}$ exists).

- **Rank Deficient:** If the rank is less than the dimension, the matrix maps the entire input space onto a lower-dimensional **sub-space** in the output.

- **Row Space vs. Null Space:** Any input vector $x$ can be **decomposed** into a component in the **row space** (which maps the output) and a component in the **null space** (which maps to zero).

- **Column Space (Range):** The set of all possible outputs $Ax$ that can be reached; it is the span of the columns of $A$.

----

### Projections and Least Squares

**Projection** involves finding the point $v$ in a subspace that is closest to a given vector $y$.

• **Orthogonality:** The line connecting the original point to its projection is always **perpendicular (orthogonal)** to the subspace.

• **Least Squares:** In machine learning, we often try to solve $Ax=b$ where $b$ is outside the column space of $A$ (unreachable). We find the best $x$ by projecting b onto the column space of $A$, leading to the **normal equations**: $x=(A^TA)^{−1}A^Tb$.

----

### Eigenvalues and Eigenvectors

For a square matrix $A$, **eigenvectors** are directions that do not change their orientation when transformed by $A$, only their magnitude (scaled by the **eigenvalue** $λ$).

• **Geometric Transformation:** If you transform a unit sphere using matrix $A$, it becomes an **ellipsoid**. The axes of this ellipsoid are the eigenvectors, and their lengths are the eigenvalues.

• **Rank Correlation:** The **rank** of a matrix is equal to the number of its **non-zero eigenvalues**.

• **Symmetric Matrices:** For symmetric $A$, eigenvalues are always real, and eigenvectors are orthogonal.
