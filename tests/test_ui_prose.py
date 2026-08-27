"""Dashboard copy must follow polarity_T, not a hardcoded sarcasm script."""

from explain.ui import _apply_polarity_contract, _prediction_prose


def test_same_sign_display_is_plain_positive():
    label, _conf = _apply_polarity_contract(
        "positive_sarcasm",
        0.499,
        {
            "Dsem": 0.25,
            "Dsen": 0.76,
            "Fvt": 0.27,
            "cos_TI": 0.18,
            "polarity_T": 0.2333,
            "polarity_T_hat": 0.9977,
        },
    )
    assert label == "positive"
    html = _prediction_prose(label, 0.6, polarity_T=0.23, polarity_T_hat=0.99)
    assert "کنایه" not in html
    assert "مثبت" in html


def test_positive_sarcasm_prose_says_negative_when_text_is_negative():
    html = _prediction_prose("positive_sarcasm", 0.9, polarity_T=-0.75, polarity_T_hat=0.99)
    assert "منفی" in html
    assert "اما" in html
