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

    expanded_class_weight = []
    for k in range(n_outputs):

        y_full = y[:, k]
        classes_full = np.unique(y_full)
        classes_missing = None

        if class_weight == 'balanced' or n_outputs == 1:
            class_weight_k = class_weight
        else:
            class_weight_k = class_weight[k]

        if indices is not None:
            # Get class weights for the subsample, covering all classes in
            # case some labels that were present in the original data are
            # missing from the sample.
            y_subsample = y[indices, k]
            classes_subsample = np.unique(y_subsample)

            weight_k = np.take(compute_class_weight(class_weight_k,
                                                    classes=classes_subsample,
                                                    y=y_subsample),
                               np.searchsorted(classes_subsample,
                                               classes_full),
                               mode='clip')

            classes_missing = set(classes_full) - set(classes_subsample)
        else:
            weight_k = compute_class_weight(class_weight_k,
                                            classes=classes_full,
                                            y=y_full)

        weight_k = weight_k[np.searchsorted(classes_full, y_full)]

        if classes_missing:
            # Make missing classes' weight zero
            weight_k[np.in1d(y_full, list(classes_missing))] = 0.

        expanded_class_weight.append(weight_k)

    expanded_class_weight = np.prod(expanded_class_weight,
                                    axis=0,
                                    dtype=np.float64)

    return expanded_class_weight