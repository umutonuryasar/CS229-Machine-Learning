## Lecture 03

### Probability Theory, Statistics, and Introduction to Linear Regression

#### 1. Completion of Probability Theory Review

Lecture 3 concludes the review of probability concepts necessary for machine learning, focusing on continuous variables, expectations, and complex distributions.

• **Continuous Random Variables:** Unlike discrete variables, the probability of a continuous random variable taking a specific value is always zero (P(X=x)=0). Instead, probabilities are defined over **intervals**, represented by the area under the **Probability Density Function (PDF)**.

• **Expectation ($E[X]$):** Informally, this is the average value a random variable takes.

- **Discrete:** Weighted sum of values multiplied by their probabilities.

- **Continuous:** The integral of the variable multiplied by its density.

- **Monte Carlo Estimate:** Expectation can be approximated by averaging a large number of random samples ($n→∞$), a principle known as the **Law of Large Numbers**.

• **Variance:** Measures how spread out a distribution is around its mean.

• **Joint and Marginal Distributions:** A joint distribution $P(X,Y)$ captures information about two variables simultaneously. **Marginalization** is the process of summing (discrete) or integrating (continuous) out one variable to find the distribution of the other.

• **Bayes' Theorem:** Relates conditional, joint, and marginal probabilities: 
	
						$P(Y∣X)=P(X)P(X∣Y)P(Y)​$.

• **Independence:** Two random variables are independent if their joint distribution is the product of their marginals: $P(X,Y)=P(X)P(Y)$.

--------------------------------------------------------------------------------

#### 2. The Multivariate Gaussian Distribution

This distribution is fundamental to many machine learning models. It is defined by two parameters:

• **Mean vector ($μ$):** Determines the center of the distribution.

• **Covariance matrix ($Σ$):** Determines the shape and orientation of the "bell curve". $Σ$ must be a **positive semi-definite** symmetric matrix.

The **Quadratic Form** $(x−μ)^TΣ^{−1}(x−μ)$ found within the Gaussian PDF determines the distance of a point from the mean, weighted by the covariance.

--------------------------------------------------------------------------------

#### 3. Statistics vs. Machine Learning

The sources distinguish these fields based on their primary goals:

• **Probability:** Makes statements about **observations** given fixed parameters.

• **Statistics:** Given **data**, makes statements or inferences about **parameters** (e.g., confidence intervals, p-values).

• **Machine Learning:** Uses statistical tools to learn a model from **training data** with the ultimate goal of making **predictions** on future, unseen data.

--------------------------------------------------------------------------------

#### 4. Maximum Likelihood Estimation (MLE)

MLE is a standard procedure for estimating parameters ($θ$) from data.

• **Likelihood vs. Probability:** While the mathematical expression is often similar, **probability** treats parameters as fixed and data as variable, while **likelihood** treats data as fixed and parameters as the variables to be optimized.

• **IID Assumption:** We typically assume data points are **Independently and Identically Distributed**, allowing us to calculate the joint likelihood as a product of individual likelihoods.

• **Log-Likelihood:** Because the logarithm is a monotonically increasing function, maximizing the **log-likelihood** yields the same parameter estimates as maximizing the likelihood directly, but simplifies the math (converting products into sums).

**MLE for Multivariate Gaussian:** By taking the derivative of the log-likelihood with respect to $μ$ and $Σ$ and setting them to zero, we derive:

• $μ_{MLE}$ ​= The average of the observed data points.

• $Σ_{MLE}$​ = The empirical covariance of the data points.

--------------------------------------------------------------------------------

#### 5. Introduction to Linear Regression

Linear regression is a **supervised learning** problem where the goal is to learn the relationship between input vectors ($x$) and real-valued outputs ($y$).

• **Training Set:** A collection of n pairs ($x(i)$,$y(i)$).

• **Hypothesis Function ($hθ​$):** An algorithm parameterized by $θ$ that takes a new $x$ and predicts a corresponding $y$.