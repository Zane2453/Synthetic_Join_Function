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

    if m > n:
        m, n = n, m

    indptr = X.indptr
    m_start = indptr[m]
    m_stop = indptr[m + 1]
    n_start = indptr[n]
    n_stop = indptr[n + 1]
    nz_m = m_stop - m_start
    nz_n = n_stop - n_start

    if nz_m != nz_n:
        # Modify indptr first
        X.indptr[m + 2:n] += nz_n - nz_m
        X.indptr[m + 1] = m_start + nz_n
        X.indptr[n] = n_stop - nz_m

    X.indices = np.concatenate([X.indices[:m_start],
                                X.indices[n_start:n_stop],
                                X.indices[m_stop:n_start],
                                X.indices[m_start:m_stop],
                                X.indices[n_stop:]])
    X.data = np.concatenate([X.data[:m_start],
                             X.data[n_start:n_stop],
                             X.data[m_stop:n_start],
                             X.data[m_start:m_stop],
                             X.data[n_stop:]])

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