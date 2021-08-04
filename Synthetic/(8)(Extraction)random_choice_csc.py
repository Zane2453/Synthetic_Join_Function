def run(classes, class_prob):
    for j in range(len(classes)):
        classes[j] = np.asarray(classes[j])
        classes[j] = classes[j].astype(np.int64, copy=False)
        if class_prob is None:
            class_prob_j = np.empty(shape=claclassesss[j].shape[0])
            class_prob_j.fill(1/classes[j].shape[0])
        else:
            class_prob_j = np.asarray(class_prob[j])
    return classes