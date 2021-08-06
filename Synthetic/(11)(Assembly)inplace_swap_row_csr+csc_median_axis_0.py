def run(X, m, n):
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

    indptr = X.indptr
    n_samples, n_features = X.shape
    median = np.zeros(n_features)

    for f_ind, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):
        # Prevent modifying X in place
        data = np.copy(X.data[start: end])
        nz = n_samples - data.size
        median[f_ind] = _get_median(data, nz)

    return median