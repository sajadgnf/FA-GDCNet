"""F1-tuned clash threshold, not Dsem accuracy."""

import numpy as np

from eval.sarcasm import pick_score_threshold


def test_pick_score_threshold_prefers_f1_over_always_negative():
    scores = np.array([-0.2, -0.1, 0.4, 0.6])
    y = np.array([0, 0, 1, 1])
    t = pick_score_threshold(scores, y, metric="f1", low=-0.5, high=0.8, n=27)
    pred = (scores >= t).astype(int)
    assert pred.tolist() == [0, 0, 1, 1]


def test_accuracy_cut_can_ignore_the_rare_class():
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    y = np.array([0, 0, 0, 1])
    t = pick_score_threshold(
        scores, y, metric="accuracy", low=0.0, high=1.0, n=21
    )
    pred = (scores >= t).astype(int)
    # An accuracy cut is allowed to predict no sarcasm if that is the majority.
    assert int(pred.sum()) <= 1
