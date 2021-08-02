def _get_median(data, n_zeros):
    n_elems = len(data) + n_zeros
    
    n_negative = np.count_nonzero(data < 0)
    middle, is_odd = divmod(n_elems, 2)
    data.sort()

    if is_odd:
        return get_elem_at_rank(middle, data, n_negative, n_zeros)

    return (get_elem_at_rank(middle - 1, data, n_negative, n_zeros) +
            get_elem_at_rank(middle, data, n_negative, n_zeros)) / 2.