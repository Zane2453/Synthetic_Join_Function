def run(class_weight, y, *, indices=None):
    for k in range(n_outputs):
        y_full = y[:, k]
        classes_full = np.unique(y_full)
        classes_missing = None

        if class_weight == 'balanced' or n_outputs == 1:
            class_weight_k = class_weight
        else:
            class_weight_k = class_weight[k]

    return class_weight