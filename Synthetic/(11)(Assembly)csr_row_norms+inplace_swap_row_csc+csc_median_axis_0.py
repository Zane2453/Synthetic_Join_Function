def run(X, m, n):
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)
    if m < 0:
        m += X.shape[0]
    if n < 0:
        n += X.shape[0]

    m_mask = X.indices == m
    X.indices[X.indices == n] = m
    X.indices[m_mask] = n

    indptr = X.indptr
    n_samples, n_features = X.shape
    median = np.zeros(n_features)

    for f_ind, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):
        # Prevent modifying X in place
        data = np.copy(X.data[start: end])
        nz = n_samples - data.size
        median[f_ind] = _get_median(data, nz)

    return median