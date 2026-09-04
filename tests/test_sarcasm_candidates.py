"""Tests for sarcasm caption heuristics."""

from data.sarcasm_candidates import (
    is_news_or_meta_caption,
    is_political_caption,
    is_sarcasm_candidate_caption,
)


def test_sad_text_happy_emoji() -> None:
    assert is_sarcasm_candidate_caption("من خیلی ناراحتم 😂😊")


def test_contrast_clause() -> None:
    assert is_sarcasm_candidate_caption("حالم بده ولی دارم میخندم")


def test_plain_selfie_not_candidate() -> None:
    assert not is_sarcasm_candidate_caption("سلفی امروز #سلفی #روزمره")


def test_irony_hashtag_alone_is_not_candidate() -> None:
    assert not is_sarcasm_candidate_caption("یکمی طنز برای شب #طنز")
    assert not is_sarcasm_candidate_caption("حق #تیکه_دار #کنایه #متن_خاص")


def test_news_about_sarcasm_is_not_candidate() -> None:
    news = "استوری کنایه‌آمیز صدف خطاب به الناز ملک وایرال شد"
    assert is_news_or_meta_caption(news)
    assert not is_sarcasm_candidate_caption(news)


def test_bitter_cue_is_candidate() -> None:
    assert is_sarcasm_candidate_caption("حال ما خوب است اما تو باور نکن")


def test_strict_mode_drops_quote_only() -> None:
    quoted = 'گفت «سلام» و رفت #سلفی #روزمره امروز'
    assert is_sarcasm_candidate_caption(quoted) is False
    assert is_sarcasm_candidate_caption(quoted, allow_weak_cues=False) is False


def test_political_iran_captions_are_dropped() -> None:
    assert is_political_caption(
        "طنز تلخ قطعی برق در ایران! خمینی وعده آب و برق مجانی داده بود #خمینی"
    )
    assert is_political_caption(
        "#حسن_ریوندی #رییسی #وزیر #مسئولین_بی_کفایت #طنزسیاسی"
    )
    assert is_political_caption("از نظر صدا و سیما همه بدبختن ما خوبیم😂")
    assert not is_political_caption(
        "خانوم صادقی چیکار میکنید؟😂 #دابسمش_ایرانی #طنز_ایرانی"
    )
    assert not is_political_caption(
        "ما که اسم کسی رو نیاوردیم… #تیکه_دار #طنز #ایران"
    )
