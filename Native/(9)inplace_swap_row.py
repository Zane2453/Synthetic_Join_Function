def inplace_swap_row(X, m, n):
    if isinstance(X, sp.csc_matrix):
        inplace_swap_row_csc(X, m, n)
    elif isinstance(X, sp.csr_matrix):
        inplace_swap_row_csr(X, m, n)
    else:
        _raise_typeerror(X)