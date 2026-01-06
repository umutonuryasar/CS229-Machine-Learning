## Lecture 04

## Linear Regression

### 1. Overview of Supervised Learning
The goal is to learn a hypothesis function $h: \mathcal{X} \to \mathcal{Y}$ such that $h(x)$ is a reliable predictor for $y$.

* **Regression:** $y$ is continuous (e.g., price estimation).
* **Classification:** $y$ takes discrete values (classes).
* **Notation:** * $x^{(i)}$: Input features (i-th example).
    * $y^{(i)}$: Target variable.
    * $(x^{(i)}, y^{(i)})$: Training example.

---

### 2. Linear Regression Model
Approximating $y$ as a linear combination of features.

* **Hypothesis:** $h_\theta(x) = \theta_0 + \theta_1 x_1 + \dots + \theta_d x_d$
* **Vector Notation:** With intercept term $x_0 = 1$:
    $$h(x) = \sum_{i=0}^{d} \theta_i x_i = \theta^T x$$
* **Cost Function (MSE):** $$J(\theta) = \frac{1}{2} \sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)})^2$$

---

### 3. The LMS Algorithm (Gradient Descent)
Iterative optimization to minimize $J(\theta)$.

* **Update Rule:** $\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$
* **Widrow-Hoff Rule:** For a single example:
    $$\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$$
* **Batch GD:** Uses all $n$ examples per step. Guaranteed convergence to global minimum (convex $J$).
* **SGD (Stochastic):** Updates per single example. Faster for large-scale datasets.

---

### 4. The Normal Equations
Closed-form solution using Matrix Calculus.

* **Design Matrix $X$:** Each row is a training example.
* **Matrix Form J:** $J(\theta) = \frac{1}{2}(X\theta - \vec{y})^T(X\theta - \vec{y})$
* **Normal Equation:** $X^T X \theta = X^T \vec{y}$
* **Solution:** $\theta = (X^T X)^{-1} X^T \vec{y}$
* **Geometry:** Projection of $\vec{y}$ onto the column space of $X$.

---

### 5. Probabilistic Interpretation
Least-squares is equivalent to **Maximum Likelihood Estimation (MLE)** under Gaussian assumptions.

* **Assumption:** $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$, where $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ (IID noise).
* Maximizing log-likelihood $\ell(\theta)$ leads directly to minimizing $J(\theta)$.

---

### 6. Locally Weighted Linear Regression (LWR)
A non-parametric algorithm where parameters grow with data.

* **Weighting Function:** $w^{(i)} = \exp\left(-\frac{(x^{(i)} - x)^2}{2\tau^2}\right)$
* **Mechanism:** Fits a local $\theta$ for each query point $x$.
* **Bandwidth ($\tau$):** Controls the rate at which weights fall off with distance.