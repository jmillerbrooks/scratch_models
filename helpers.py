def test_results(preds_a, preds_b, tolerance=0.0001):
    # takes array-like of two predictions of same length,
    # tests each pair of corresponding elements is within
    # +/- value passed in tolerance
    result_vec = np.abs(preds_a - preds_b) < tolerance
    if np.all(result_vec):
        return True, None
    else:
        # return False and list of indices where test fails
        return False, [(a, b) for a, b in zip(np.where(result_vec == False)[0], np.where(result_vec == False)[1])]
