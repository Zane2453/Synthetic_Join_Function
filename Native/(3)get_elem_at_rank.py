def get_elem_at_rank(rank, n_negative, n_zeros):
    if rank < n_negative:
        return rank
    if rank - n_negative < n_zeros:
        return 0
    return rank - n_negative