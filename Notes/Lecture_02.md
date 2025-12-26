### I. Linear Algebra Review (Continued)

- **Determinants and Volume:** Geometrically, the absolute value of a determinant represents the **ratio of the volume of an output shape to the volume of an input shape** after a linear transformation. If a matrix is not full rank, at least one eigenvalue is zero, causing the shape to collapse and the volume (and determinant) to become zero.

-  **The Spectrum and Spectral Theorem:** The **spectrum** is the collection of a matrix's eigenvalues. The **Spectral Theorem** states that every real symmetric square matrix has real-valued eigenvalues and orthonormal eigenvectors.

- **Quadratic Forms and Definiteness:** A quadratic form is expressed as $x^TAx$. The **definiteness** of a symmetric matrix $A$ (Positive Definite, Positive Semidefinite, etc.) is determined by whether $x^TAx$ is always positive, non-negative, or otherwise for all non-zero x. This property is directly linked to the signs of the matrix's eigenvalues.

- **Matrix Decompositions:**

	-  **Singular Value Decomposition (SVD):** Decomposes any matrix into $UΣVT$, where U and $V$ are orthonormal and $Σ$ contains real-valued singular values.

	- **Eigenvalue Decomposition:** Decomposes a square matrix into $UDU^{−1}$. For square symmetric matrices, SVD and eigenvalue decomposition are identical.

### II. Matrix Calculus

• **Gradients and Hessians:**

- The **Gradient** (∇) is the vector of first-order partial derivatives of a scalar-valued function with respect to a vector input, representing the direction of steepest ascent.

- The **Hessian** is a symmetric square matrix of second-order partial derivatives. A **positive semi-definite Hessian** indicates the function is **convex** (bowl-shaped).

• **Jacobians:** These are matrices of first-order partial derivatives for functions that map a vector input to a vector output.

• **Key Identities:** One frequently used identity in machine learning (particularly with Gaussians) is that the gradient of the log-determinant, $∇A​log∣A∣$, equals $A^{−1}$.

### III. Probability Theory

• **Foundations:** Probability involves a **sample space** (all possible outcomes), **events** (subsets of outcomes), and a **probability measure** that assigns a value between 0 and 1 to those events.

• **Random Variables:** A random variable is technically a **function** that maps outcomes from the sample space to real numbers.

• **Distributions:**
- **Cumulative Distribution Function (CDF):** The probability that a random variable X is less than or equal to a value t.

- **Probability Mass Function (PMF) / Density Function (PDF):** These describe the probability of discrete values or the relative likelihood of continuous values, respectively.

• **Expected Value:** The expectation of a function g(x) is the **weighted average** of the function's values, weighted by the probability density of the input.

• **Law of Large Numbers:** This principle states that the average of many independent samples (the **Monte Carlo estimate**) converges to the true expected value as the number of samples increases to infinity.

----
**Analogy for Matrix Transformations:** Think of a linear transformation as a **funhouse mirror**. The **eigenvectors** are the specific directions in front of the mirror where your reflection stays perfectly upright or horizontal, only getting taller or shorter. The **eigenvalues** tell you exactly how much you are being stretched or shrunk in those specific directions. If an eigenvalue is zero, that dimension of your reflection vanishes entirely, like a 3D object being squashed into a 2D shadow.
