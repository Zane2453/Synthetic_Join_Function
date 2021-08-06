def run(X, copy=True):
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X