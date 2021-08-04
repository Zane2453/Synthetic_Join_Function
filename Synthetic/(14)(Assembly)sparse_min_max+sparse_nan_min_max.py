def run(X, axis, ignore_nan):
    if ignore_nan:
       return sparse_nan_min_max(X, axis=axis)
    else:
       return sparse_min_max(X, axis=axis)