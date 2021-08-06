def run(X, min_value=0):
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)
    min_ = X.min()
    if min_ < min_value:
        if sparse.issparse(X):
            raise ValueError("Cannot make the data matrix"
                             " nonnegative because it is sparse."
                             " Adding a value to every entry would"
                             " make it no longer sparse.")
        X = X + (min_value - min_)
    return X