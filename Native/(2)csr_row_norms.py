def csr_row_norms(X):
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)
    return _csr_row_norms(X.data, X.shape, X.indices, X.indptr)