def compute_sample_weight(class_weight, y, *, indices=None):
    y = np.atleast_1d(y)
    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))
    n_outputs = y.shape[1]

    if isinstance(class_weight, str):
        if class_weight not in ['balanced']:
            raise ValueError('The only valid preset for class_weight is '
                             '"balanced". Given "%s".' % class_weight)
    elif (indices is not None and
          not isinstance(class_weight, str)):
        raise ValueError('The only valid class_weight for subsampling is '
                         '"balanced". Given "%s".' % class_weight)
    elif n_outputs > 1:
        if (not hasattr(class_weight, "__iter__") or
                isinstance(class_weight, dict)):
            raise ValueError("For multi-output, class_weight should be a "
                             "list of dicts, or a valid string.")
        if len(class_weight) != n_outputs:
            raise ValueError("For multi-output, number of elements in "
                             "class_weight should match number of outputs.")