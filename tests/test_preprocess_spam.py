"""Spam caption filter tests."""

from data.preprocess import is_spam_caption, preprocess_caption


def test_spam_detects_ai_prompt_bait() -> None:
    cap = preprocess_caption(
        "برای دریافت پرامپت کلمه سلفی رو کامنت کن #هوش_مصنوعی #سلفی"
    )
    assert is_spam_caption(cap)


def test_spam_detects_real_daily_caption() -> None:
    cap = preprocess_caption(
        "این روزا چسبیدم به زندگی. سعی میکنم آرومتر نفس بکشم #سلفی #خانواده"
    )
    assert not is_spam_caption(cap)
