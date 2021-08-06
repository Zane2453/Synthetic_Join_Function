def run(X, last_mean, last_var, last_n, weights=None, m, n):
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)
    X_dtype = X.dtype
    if weights is None:
        weights = np.ones(X.shape[0], dtype=X_dtype)
    elif weights.dtype not in [np.float32, np.float64]:
        weights = weights.astype(np.float64, copy=False)
    if last_n.dtype not in [np.float32, np.float64]:
        last_n = last_n.astype(np.float64, copy=False)

    if m < 0:
        m += X.shape[0]
    if n < 0:
        n += X.shape[0]

    m_mask = X.indices == m
    X.indices[X.indices == n] = m
    X.indices[m_mask] = n

    return (X.data,
            np.sum(weights),
            X.shape[1],
            X.indices,
            X.indptr,
            X.format,
            last_mean.astype(X_dtype, copy=False),
            last_var.astype(X_dtype, copy=False),
            last_n.astype(X_dtype, copy=False),
            weights.astype(X_dtype, copy=False))