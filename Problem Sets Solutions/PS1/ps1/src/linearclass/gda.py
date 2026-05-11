import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    util.plot(x_valid, y_valid, clf.theta, save_path.replace('.txt', '.png'))

    predictions = clf.predict(x_valid)
    np.savetxt(save_path, predictions)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape

        phi = np.mean(y == 1)
        mu_0 = x[y == 0].mean(axis=0)
        mu_1 = x[y == 1].mean(axis=0)

        diff = x.copy()
        diff[y == 0] -= mu_0
        diff[y == 1] -= mu_1
        sigma = (diff.T @ diff) / n

        sigma_inv = np.linalg.inv(sigma)
        theta_body = sigma_inv @ (mu_1 - mu_0)
        theta_0 = 0.5 * (mu_0 @ sigma_inv @ mu_0 - mu_1 @ sigma_inv @ mu_1) + np.log(phi / (1 - phi))

        # Store as [theta_0, theta_1, theta_2] for use with util.plot
        self.theta = np.concatenate([[theta_0], theta_body])
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-(x @ self.theta[1:] + self.theta[0])))
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
