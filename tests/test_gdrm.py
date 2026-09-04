"""Tests for the GDRM feature extraction (pure numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from inference.gdrm import (
    DEFAULT_FVT_THRESHOLD,
    FEATURE_NAMES,
    build_feature_vector,
    compute_dsem,
    compute_dsen,
    compute_fvt,
    cosine_distance,
    cosine_similarity,
    polarity_l1,
    polarity_scalar,
)


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_norm_returns_zero():
    a = np.zeros(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_all_nan_inputs_return_zero():
    a = np.array([np.nan, np.nan])
    b = np.array([1.0, 2.0])
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_cosine_distance_inverse_of_similarity():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine_distance(a, b) == pytest.approx(1.0)


def test_polarity_l1_basic():
    a = np.array([0.7, 0.3])
    b = np.array([0.2, 0.8])
    assert polarity_l1(a, b) == pytest.approx(1.0)


def test_polarity_scalar_range():
    assert polarity_scalar(np.array([1.0, 0.0])) == pytest.approx(-1.0)
    assert polarity_scalar(np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert polarity_scalar(np.array([0.5, 0.5])) == pytest.approx(0.0)


def test_canonicalize_snappfood_happy_sad_order():
    """Snappfood head is HAPPY=0, SAD=1 — must become (p_neg, p_pos)."""
    from inference.models import _canonicalize_polarity_probs

    class _M:
        config = type("C", (), {"id2label": {0: "HAPPY", 1: "SAD"}})()

    # Model is confident SAD → ordered [p_sad, p_happy] → negative scalar
    ordered = _canonicalize_polarity_probs(np.array([0.1, 0.9], dtype=np.float32), _M())
    assert ordered[0] == pytest.approx(0.9)  # neg
    assert ordered[1] == pytest.approx(0.1)  # pos
    assert polarity_scalar(ordered) == pytest.approx(-0.8)


def test_canonicalize_three_class_head_drops_neutral():
    """A {negative, neutral, positive} head maps to (p_neg, p_pos)."""
    from inference.models import _canonicalize_polarity_probs

    class _M:
        config = type(
            "C", (), {"id2label": {0: "negative", 1: "neutral", 2: "positive"}}
        )()

    ordered = _canonicalize_polarity_probs(
        np.array([0.7, 0.2, 0.1], dtype=np.float32), _M()
    )
    assert ordered[0] == pytest.approx(0.7)
    assert ordered[1] == pytest.approx(0.1)
    assert polarity_scalar(ordered) == pytest.approx(-0.6)


def test_scalar_to_probs_round_trips_through_polarity_scalar():
    from inference.models import _scalar_to_probs

    for s in (-1.0, -0.42, 0.0, 0.42, 1.0):
        assert polarity_scalar(_scalar_to_probs(s)) == pytest.approx(s)


def test_negation_detection_uses_function_words():
    from inference.models import has_negation

    assert has_negation("ناراحت نیستم")
    assert has_negation("بد نبود")
    assert has_negation("مشکلی نیست")
    assert has_negation("کاش نمی‌رفت")
    # No negation cue — must not trigger on ordinary vocabulary.
    assert not has_negation("ناراحت است")
    assert not has_negation("خوشحالم")
    assert not has_negation("به درک")
    assert not has_negation("نه تنها خوشحالم")
    assert has_negation("نه")
    assert has_negation("بدون امید")


def test_denegate_rewrites_to_affirmative_claim():
    from inference.models import denegate

    assert "هستم" in denegate("ناراحت نیستم")
    assert "بود" in denegate("بد نبود")
    assert "نمی" not in denegate("کاش نمی‌رفت")


def test_contrast_tail_picks_final_clause():
    from inference.models import contrast_tail

    assert contrast_tail("ناراحتی بود اما الان خوشحال است") == "الان خوشحال است"
    assert contrast_tail("روز بدی بود ولی شب خوبی داشتم") == "شب خوبی داشتم"
    assert contrast_tail("اول ناراحت بودم اما بعد خندیدم ولی الان آرامم") == "الان آرامم"
    # No contrast marker → whole text is the clause.
    assert contrast_tail("خوشحالم") == "خوشحالم"


def test_polarity_probs_flips_negated_clause(monkeypatch):
    """«ناراحت نیستم» must not inherit the negative score of «ناراحت»."""
    from inference import models as M

    def fake_head(_bundle, text: str) -> float:
        return -0.9 if "ناراحت" in text else 0.0

    monkeypatch.setattr(M, "_head_scalar", fake_head)
    probs = M.polarity_probs(object(), "ناراحت نیستم")
    assert polarity_scalar(probs) > 0.5


def test_polarity_probs_keeps_raw_score_when_denegate_is_noop(monkeypatch):
    """Captions with no rewriteable cue must not be sign-flipped."""
    from inference import models as M

    monkeypatch.setattr(M, "_head_scalar", lambda _b, _t: 0.4)
    probs = M.polarity_probs(object(), "امروز هوا ابری است")
    assert polarity_scalar(probs) == pytest.approx(0.4)


def test_mourning_formula_overrides_false_positive_head(monkeypatch):
    """«شادروان» must not be scored positive because it contains «شاد»."""
    from inference import models as M

    monkeypatch.setattr(M, "_head_scalar", lambda _b, _t: 0.47)
    for caption in (
        "شادروان",
        "مرحوم پدرم",
        "فقید",
        "انا لله",
        "خدا بیامرزدش",
        "روحشون شاد",
        "به درک",
        "گور پدرش",
        "برو بمیر",
        "RIP",
        "حال ما خوب است اما تو باور نکن",
        "زنده‌یاد مادر",
        "آسمانی شد",
        "غم آخرتان باشد",
        "یادش گرامی",
        "condolences",
        "به رحمت ایزدی پیوست",
        "هر هر هر خنده داره؟",
        "هرهرهر",
    ):
        assert polarity_scalar(M.polarity_probs(object(), caption)) < -0.5
    # Must not fire on ordinary «درک» (reading comprehension).
    assert polarity_scalar(M.polarity_probs(object(), "درک مطلب سخت است")) == pytest.approx(0.47)
    # Bare «خنده داره؟» is the idiom, not the mocking laugh.
    assert polarity_scalar(M.polarity_probs(object(), "خنده داره؟")) == pytest.approx(0.47)


def test_polarity_sends_khandun_as_standard_spelling(monkeypatch):
    from inference import models as M

    seen: list[str] = []

    def fake_head(_bundle, text: str) -> float:
        seen.append(text)
        return 0.4

    monkeypatch.setattr(M, "_head_scalar", fake_head)
    M.polarity_probs(object(), "نبین که خندونم کلا ادم خندونیم")
    assert seen
    blob = " ".join(seen)
    assert "خندون" not in blob
    assert "نگاه نکن که" in blob
    assert "میخندم" in blob
    assert "خندانی هستم" in blob


def test_polarity_rewrites_khande_dare_idiom(monkeypatch):
    from inference import models as M

    seen: list[str] = []
    monkeypatch.setattr(M, "_head_scalar", lambda _b, t: seen.append(t) or 0.4)
    M.polarity_probs(object(), "خنده داره؟")
    assert seen
    assert "بامزه است" in seen[0]
    assert "خنده داره" not in seen[0]


def test_polarity_probs_weights_contrast_tail(monkeypatch):
    """The clause after «اما» dominates the caption score."""
    from inference import models as M

    def fake_head(_bundle, text: str) -> float:
        # Tail alone is positive; the whole caption reads negative.
        return 0.9 if text.strip() == "الان خوشحال است" else -0.6

    monkeypatch.setattr(M, "_head_scalar", fake_head)
    probs = M.polarity_probs(object(), "ناراحتی بود اما الان خوشحال است")
    assert polarity_scalar(probs) > 0.3


def test_polarity_probs_handles_double_contrast(monkeypatch):
    """The *last* اما/ولی clause is the asserted one."""
    from inference import models as M

    def fake_head(_bundle, text: str) -> float:
        t = text.strip()
        if t == "الان آرامم":
            return 0.8
        if t == "اول ناراحت بودم اما بعد خندیدم ولی الان آرامم":
            return -0.4
        return -0.2

    monkeypatch.setattr(M, "_head_scalar", fake_head)
    probs = M.polarity_probs(object(), "اول ناراحت بودم اما بعد خندیدم ولی الان آرامم")
    assert polarity_scalar(probs) > 0.4


def test_compute_dsem_contradiction():
    """A clearly contradicting `T_hat` produces a large Dsem."""
    T_emb = np.array([1.0, 0.0, 0.0])
    T_hat_emb = np.array([0.0, 0.0, 1.0])
    assert compute_dsem(T_emb, T_hat_emb) == pytest.approx(1.0)


def test_compute_dsem_agreement():
    v = np.array([1.0, 2.0, 3.0])
    assert compute_dsem(v, v) == pytest.approx(0.0)


def test_compute_dsen_polarity_contradiction():
    pos_in_T = np.array([0.05, 0.95])
    neg_in_T_hat = np.array([0.95, 0.05])
    # L1 = |0.05-0.95| + |0.95-0.05| = 1.8
    assert compute_dsen(pos_in_T, neg_in_T_hat) == pytest.approx(1.8)


def test_compute_fvt_perfect_match():
    v = np.array([1.0, 0.0])
    assert compute_fvt(v, v) == pytest.approx(1.0)


def test_build_feature_vector_shape_and_names():
    T = np.array([1.0, 0.0, 0.0])
    Th = np.array([1.0, 0.0, 0.0])
    I = np.array([0.0, 1.0, 0.0])
    pT = np.array([0.2, 0.8])
    pTh = np.array([0.2, 0.8])

    f = build_feature_vector(
        text_emb_T=T,
        text_emb_T_hat=Th,
        image_emb_I=I,
        polarity_probs_T=pT,
        polarity_probs_T_hat=pTh,
    )
    arr = f.as_array()
    assert arr.shape == (len(FEATURE_NAMES),)
    d = f.as_dict()
    assert tuple(d.keys()) == FEATURE_NAMES
    assert f.clash == pytest.approx(-f.polarity_T * f.polarity_T_hat)


def test_build_feature_vector_sarcasm_signature():
    """Sarcasm: positive Persian caption, but image content negative.

    Dsem high (caption embedding ≠ T_hat embedding), polarity scalar of T high,
    polarity scalar of T_hat low → Dsen also high. Fvt should still be high
    (T_hat does describe the image well).
    """
    T_emb = np.array([1.0, 0.0])
    T_hat_emb = np.array([-1.0, 0.0])  # opposite direction in shared space
    I_emb = np.array([-1.0, 0.0])
    p_T = np.array([0.05, 0.95])       # positive
    p_T_hat = np.array([0.92, 0.08])   # negative

    f = build_feature_vector(
        text_emb_T=T_emb,
        text_emb_T_hat=T_hat_emb,
        image_emb_I=I_emb,
        polarity_probs_T=p_T,
        polarity_probs_T_hat=p_T_hat,
    )
    assert f.Dsem > 0.9            # large semantic gap
    assert f.Dsen > 1.5            # large polarity flip
    assert f.Fvt > 0.9             # but description still describes the image
    assert f.polarity_T > 0.5
    assert f.polarity_T_hat < -0.5
    assert f.clash > 0.4


def test_clip_style_image_polarity_moves_dsen():
    """Smile vs sad CLIP vectors must not collapse Dsen the way bland T̂ did."""
    from data.image_affect import polarity_vector

    sad_text = np.array([0.8, 0.2])
    smile = np.array(polarity_vector(0.95, 0.05))
    f = build_feature_vector(
        text_emb_T=np.array([1.0, 0.0]),
        text_emb_T_hat=np.array([1.0, 0.0]),
        image_emb_I=np.array([1.0, 0.0]),
        polarity_probs_T=sad_text,
        polarity_probs_T_hat=smile,
    )
    assert f.polarity_T < 0
    assert f.polarity_T_hat > 0.8
    assert f.Dsen > 1.0


def test_default_fvt_threshold_is_documented():
    assert 0.0 < DEFAULT_FVT_THRESHOLD < 1.0


def test_compute_clash_positive_when_polarities_oppose():
    from inference.gdrm import compute_clash, with_clash_column

    assert compute_clash(0.8, -0.7) == pytest.approx(0.56)
    assert compute_clash(0.8, 0.7) == pytest.approx(-0.56)
    six = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, -0.4]], dtype=np.float32)
    seven = with_clash_column(six)
    assert seven.shape == (1, 7)
    assert seven[0, 6] == pytest.approx(0.2)

