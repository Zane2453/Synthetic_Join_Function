def sparse_min_max(X, axis):
        return (sparse_min_or_max(X, axis, np.minimum),
                sparse_min_or_max(X, axis, np.maximum))