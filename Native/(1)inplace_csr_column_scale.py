def inplace_csr_column_scale(X, scale):
    assert scale.shape[0] == X.shape[1]
    X.data *= scale.take(X.indices, mode='clip')