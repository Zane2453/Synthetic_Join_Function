def inplace_swap_row_csc(X_shape, X_indice, m, n):
    if m < 0:
        m += X_shape
    if n < 0:
        n += X_shape

    m_mask = X_indice == m
    n_mask = X_indice == n
    
    return (m_mask, n_mask)