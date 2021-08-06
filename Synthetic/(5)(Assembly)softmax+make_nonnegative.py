def run(X, copy=True, min_value=0):
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob

    min_ = X.min()
    if min_ < min_value:
        if sparse.issparse(X):
            raise ValueError("Cannot make the data matrix"
                             " nonnegative because it is sparse."
                             " Adding a value to every entry would"
                             " make it no longer sparse.")
        X = X + (min_value - min_)
    return X