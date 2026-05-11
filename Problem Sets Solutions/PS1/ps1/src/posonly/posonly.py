import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    clf_t = LogisticRegression()
    clf_t.fit(x_train, t_train)
    util.plot(x_test, t_test, clf_t.theta, output_path_true.replace('.txt', '.png'))
    np.savetxt(output_path_true, clf_t.predict(x_test))

    # Part (b): Train on y-labels and test on true labels
    x_train_y, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)

    clf_y = LogisticRegression()
    clf_y.fit(x_train_y, y_train)
    util.plot(x_test, t_test, clf_y.theta, output_path_naive.replace('.txt', '.png'))
    np.savetxt(output_path_naive, clf_y.predict(x_test))

    # Part (f): Apply correction factor using validation set and test on true labels
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    alpha = np.mean(clf_y.predict(x_valid)[y_valid == 1])

    preds_adjusted = clf_y.predict(x_test) / alpha
    np.savetxt(output_path_adjusted, preds_adjusted)
    util.plot(x_test, t_test, clf_y.theta, output_path_adjusted.replace('.txt', '.png'), correction=alpha)
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
