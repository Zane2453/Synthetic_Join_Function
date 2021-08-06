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

    return X