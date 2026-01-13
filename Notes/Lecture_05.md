## Lecture 05

## Classification, Perceptron, and Logistic Regression

### 1. Overview and Recap

The fifth lecture transitions from **regression** (predicting continuous values) to **classification**, where the goal is to predict discrete labels $y \in \{0, 1\}$. The lecture recaps that while linear regression uses the **Ordinary Least Squares** cost function to find a signal amidst Gaussian noise, classification requires different probabilistic models and hypothesis functions.

---

### 2. The Perceptron Algorithm

The **Perceptron** is a simple, streaming classification algorithm of historical interest.
- **Hypothesis Function:** $h_\theta(x) = g(\theta^T x)$, where $g(z)$ is a **threshold function** that outputs $1$ if $z \ge 0$ and $0$ if $z < 0$.
- **Update Rule:** $\theta := \theta + \alpha(y^{(i)} - h_\theta(x^{(i)}))x^{(i)}$.
- **Intuition:** If the model misclassifies a positive example ($y=1, h=0$), it adds a fraction of the input vector $x$ to $\theta$ to increase the dot product; if it misclassifies a negative example ($y=0, h=1$), it subtracts $x$ to decrease the dot product.
- **Convergence:** Theory proves that if the data is **linearly separable**, the Perceptron will eventually find a separating hyperplane.

---
### 3. Logistic Regression

Considered the "workhorse of machine learning," **Logistic Regression** is a discriminative algorithm used widely in production.
- **Hypothesis Function:** $h_\theta(x) = g(\theta^T x)$, where $g(z) = \frac{1}{1 + e^{-z}}$ is the **Sigmoid (Logistic) function**.
- **Probabilistic Interpretation:** The output $h_\theta(x)$ is treated as the probability $P(y=1|x; \theta)$. The model assumes a **Bernoulli distribution** for the labels.
- **Maximizing Likelihood:** The parameters $\theta$ are fit by maximizing the **log-likelihood** $\ell(\theta)$, which is the sum of $(y^{(i)} \log h(x^{(i)}) + (1 - y^{(i)}) \log(1 - h(x^{(i)})))$ over the training set.
- **Gradient Ascent Update:** The resulting update rule is $\theta := \theta + \alpha(y^{(i)} - h_\theta(x^{(i)}))x^{(i)}$. Although this looks identical to the Perceptron and Linear Regression updates, $h_\theta(x)$ here is a non-linear function.

---
### 4. Newton's Method (Newton-Raphson)

**Newton's Method** provides an alternative to Gradient Descent for maximizing the log-likelihood.
- **Mechanism:** It is a **root-finding algorithm** applied to the first derivative of the loss function. It approximates the function linearly at each step and jumps to the root of that approximation.
- **Vector Update Rule:** $\theta := \theta - H^{-1} \nabla_\theta \ell(\theta)$, where $H$ is the **Hessian matrix** of second-order partial derivatives.
- **Trade-offs:** It converges much faster (fewer iterations) than Gradient Descent but is computationally expensive because each step requires inverting a $d \times d$ Hessian matrix, an $O(d^3)$ operation.

---
### 5. Functional Analysis Intuition

The lecture concludes with a perspective from **Functional Analysis**, treating functions as **infinite-dimensional vectors**.
- **Analogy:** Just as a vector has components indexed by integers, a function has values indexed by real numbers in its domain.
- **Linear Operators:** Operations like **differentiation** can be viewed as linear operators (infinite-dimensional matrices) acting on these function-vectors.
- **Importance:** This mental framework is useful for understanding advanced topics like **Kernel methods** and **Gaussian processes**.
