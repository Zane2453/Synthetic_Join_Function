def inplace_swap_row_csc(X, m, n):
    if m < 0:
        m += X.shape[0]
    if n < 0:
        n += X.shape[0]

    m_mask = X.indices == m
    X.indices[X.indices == n] = m
    X.indices[m_mask] = n
    
    return X