import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    n, dim = x.shape

    # (1) Initialize mu and sigma by splitting unlabeled data uniformly at random into K groups
    indices = np.random.permutation(n)
    groups = np.array_split(indices, K)
    mu = [np.mean(x[g], axis=0) for g in groups]
    sigma = [np.cov(x[g].T) + 1e-6 * np.eye(dim) for g in groups]

    # (2) Initialize phi to place equal probability on each Gaussian
    phi = np.ones(K) / K

    # (3) Initialize w to place equal probability on each Gaussian
    w = np.ones((n, K)) / K
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        for i in range(x.shape[0]):
            for j in range(K):
                w[i, j] = phi[j] * gaussian_pdf(x[i], mu[j], sigma[j])
            w_sum = np.sum(w[i])
            w[i] = w[i] / w_sum if w_sum > 0 else np.ones(K) / K

        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(K):
            w_j = w[:, j]
            n_j = np.sum(w_j)
            phi[j] = n_j / x.shape[0]
            mu[j] = (w_j @ x) / n_j
            diff = x - mu[j]
            sigma[j] = (w_j[:, np.newaxis] * diff).T @ diff / n_j + 1e-6 * np.eye(x.shape[1])

        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        ll = 0.0
        for i in range(x.shape[0]):
            ll_i = sum(phi[j] * gaussian_pdf(x[i], mu[j], sigma[j]) for j in range(K))
            ll += np.log(ll_i + 1e-300)
        # *** END CODE HERE ***
        it += 1

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE ***
        n = x.shape[0]
        n_tilde = x_tilde.shape[0]
        dim = x.shape[1]
        z_tilde_flat = z_tilde.squeeze().astype(int)

        # (1) E-step: Update estimates in w (only for unlabeled examples)
        for i in range(n):
            for j in range(K):
                w[i, j] = phi[j] * gaussian_pdf(x[i], mu[j], sigma[j])
            w_sum = np.sum(w[i])
            w[i] = w[i] / w_sum if w_sum > 0 else np.ones(K) / K

        # (2) M-step: Update phi, mu, sigma using both labeled and unlabeled data
        for j in range(K):
            w_j = w[:, j]
            labeled_mask = (z_tilde_flat == j)

            sum_unlabeled = np.sum(w_j)
            sum_labeled = alpha * np.sum(labeled_mask)

            phi[j] = (sum_unlabeled + sum_labeled) / (n + alpha * n_tilde)

            mu[j] = (w_j @ x + alpha * np.sum(x_tilde[labeled_mask], axis=0)) / (sum_unlabeled + sum_labeled)

            diff_u = x - mu[j]
            diff_l = x_tilde[labeled_mask] - mu[j]
            sigma[j] = (
                (w_j[:, np.newaxis] * diff_u).T @ diff_u
                + alpha * diff_l.T @ diff_l
            ) / (sum_unlabeled + sum_labeled) + 1e-6 * np.eye(dim)

        # (3) Compute log-likelihood including labeled contribution
        prev_ll = ll
        ll = 0.0
        for i in range(n):
            ll_i = sum(phi[j] * gaussian_pdf(x[i], mu[j], sigma[j]) for j in range(K))
            ll += np.log(ll_i + 1e-300)
        for i in range(n_tilde):
            j = z_tilde_flat[i]
            ll += alpha * np.log(phi[j] * gaussian_pdf(x_tilde[i], mu[j], sigma[j]) + 1e-300)
        # *** END CODE HERE ***
        it += 1

    return w


# *** START CODE HERE ***
def gaussian_pdf(x, mu, sigma):
    """Evaluate the multivariate Gaussian PDF N(x; mu, sigma)."""
    dim = x.shape[0]
    diff = x - mu
    sign, log_det = np.linalg.slogdet(sigma)
    if sign <= 0:
        return 1e-300
    log_p = -0.5 * (dim * np.log(2 * np.pi) + log_det + diff @ np.linalg.solve(sigma, diff))
    return np.exp(log_p)
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
