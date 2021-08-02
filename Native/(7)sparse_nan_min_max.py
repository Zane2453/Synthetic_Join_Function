def sparse_nan_min_max(X, axis):
    return(sparse_min_or_max(X, axis, np.fmin),
           sparse_min_or_max(X, axis, np.fmax))