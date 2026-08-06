import numpy as np

from contextcell.audits import AuditContext, evaluate_removal
from contextcell.refinement import refine_predictions


def test_target_cannot_be_removed():
    X = np.array([[0.0], [0.1], [2.0], [2.1]], dtype=float)
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    classes = np.array(["A", "B"])
    clean = refine_predictions(X, probs, classes, k=1)
    ctx = AuditContext(X, probs, classes, clean, 1, 1.0, 1.0, True)
    try:
        evaluate_removal(ctx, target_idx=0, remove_idx=[0])
    except ValueError:
        return
    raise AssertionError("Removing the target should fail.")


def test_removal_does_not_mutate_inputs():
    X = np.array([[0.0], [0.1], [2.0], [2.1]], dtype=float)
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    X_before = X.copy()
    probs_before = probs.copy()
    classes = np.array(["A", "B"])
    clean = refine_predictions(X, probs, classes, k=1)
    ctx = AuditContext(X, probs, classes, clean, 1, 1.0, 1.0, True)
    evaluate_removal(ctx, target_idx=0, remove_idx=[1])
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(probs, probs_before)


def test_zero_context_weight_is_removal_invariant():
    X = np.array([[0.0], [0.1], [2.0], [2.1]], dtype=float)
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    classes = np.array(["A", "B"])
    clean = refine_predictions(X, probs, classes, k=2, neighbor_weight=0.0)
    ctx = AuditContext(X, probs, classes, clean, 2, 1.0, 0.0, True)
    outcome = evaluate_removal(ctx, target_idx=0, remove_idx=[1, 2])
    assert not outcome.target_flipped
    assert outcome.target_label == clean.labels[0]
    assert np.isclose(outcome.target_probability_original_class, probs[0, 0])


def test_multistart_and_beam_preserve_target():
    from contextcell.audits import beam_search_removal, multistart_greedy_removal
    X = np.array([[0.0], [0.05], [0.1], [1.0], [1.1], [1.2]], dtype=float)
    probs = np.array([
        [0.51, 0.49], [0.95, 0.05], [0.90, 0.10],
        [0.05, 0.95], [0.10, 0.90], [0.15, 0.85],
    ])
    classes = np.array(["A", "B"])
    clean = refine_predictions(X, probs, classes, k=3)
    ctx = AuditContext(X, probs, classes, clean, 3, 1.0, 1.0, True)
    pool = np.array([1, 2, 3, 4, 5])
    rng = np.random.default_rng(123)
    multi, _ = multistart_greedy_removal(ctx, 0, pool, 2, 1, None, 3, rng)
    beam = beam_search_removal(ctx, 0, pool, 2, 1, None, 2, 3, np.random.default_rng(123))
    assert all(0 not in out.removed for out in multi)
    assert all(0 not in out.removed for out in beam)
