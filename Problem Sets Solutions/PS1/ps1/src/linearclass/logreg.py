import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_valid, y_valid, clf.theta, save_path.replace('.txt', '.png'))

    predictions = clf.predict(x_valid)
    np.savetxt(save_path, predictions)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape
        if self.theta is None:
            self.theta = np.zeros(d)

        for _ in range(self.max_iter):
            h = self.predict(x)                        # (n,)

            # Gradient: (1/n) X^T (h - y)
            grad = (1 / n) * x.T @ (h - y)            # (d,)

            # Hessian: (1/n) X^T diag(h*(1-h)) X
            w = h * (1 - h)                            # (n,)
            H = (1 / n) * (x * w[:, None]).T @ x      # (d, d)

            # Newton step: solve H delta = grad
            delta = np.linalg.solve(H, grad)
            self.theta -= delta

            if np.linalg.norm(delta, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x @ self.theta))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
