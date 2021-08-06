def csr_row_norms(X):
    if X.dtype not in [np.float32, np.float64]:
        X = X.astype(np.float64)
    return (X.data, X.shape, X.indices, X.indptr)