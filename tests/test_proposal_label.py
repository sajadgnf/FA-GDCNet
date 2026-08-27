"""Proposal-aligned 5-class assignment (no extra labels)."""

from data.proposal_label import assign_proposal_label, hat_affect, text_affect


def test_polarity_vector_is_neg_pos_pair():
    from data.image_affect import polarity_vector

    assert polarity_vector(0.9, 0.1) == [0.1, 0.9]


def test_hat_affect_smiling_description():
    assert hat_affect("The person is smiling and looking at the camera.") == "pos"
    assert hat_affect("The man looks sad and is crying.") == "neg"
    assert hat_affect("A person standing in front of a wall.") == "neu"


def test_negative_caption_smiling_photo_is_positive_sarcasm():
    # DZLDw8LN3J4-style: bitter inflation joke + smiling couple selfie
    label = assign_proposal_label(
        "حتی از وقتی فهمیدن میتونیم تیشرت ارزون بپوشیم اونم گرون کردن 😅 #طنز_تلخ",
        pol_T=[0.55, 0.45],
        pol_T_hat=[0.46, 0.54],
        generated="The person is smiling and looking at the phone.",
    )
    assert label == "positive_sarcasm"


def test_verbal_joke_aligned_fail_is_positive_not_sarcasm():
    # DZakR_gIkvu-style: self-deprecating joke, distressed photo, #طنز
    # Polarity may look positive because of 😅 — still not negative_sarcasm.
    label = assign_proposal_label(
        "ما بخاطر سیب بهشتو از دست دادیم… بخاطر چاقاله دیگه چیزی برای از دست دادن نداریم...😅😅 #طنز",
        pol_T=[0.20, 0.80],
        pol_T_hat=[0.60, 0.40],
        generated="A person wearing a black cap standing in front of a rough wall.",
        visual_hat="neg",
    )
    assert label == "positive"


def test_sad_text_sad_photo_is_negative():
    label = assign_proposal_label(
        "من خوردم زمین",
        pol_T=[0.80, 0.20],
        pol_T_hat=[0.70, 0.30],
        generated="A woman looking sad and crying.",
    )
    assert label == "negative"


def test_happy_text_happy_photo_is_positive():
    label = assign_proposal_label(
        "چه روز زیبایی خیلی خوشحالم",
        pol_T=[0.10, 0.90],
        pol_T_hat=[0.20, 0.80],
        generated="A man is smiling and laughing.",
    )
    assert label == "positive"


def test_cheerful_caption_gloomy_image_is_negative_sarcasm():
    label = assign_proposal_label(
        "چه روز عالی‌ای خیلی خوشحالم",
        pol_T=[0.10, 0.90],
        generated="The woman looks sad and is crying.",
    )
    assert label == "negative_sarcasm"


def test_im_not_fine_plus_smile_is_positive_sarcasm():
    assert text_affect("حالم خوب نیست") == "neg"
    label = assign_proposal_label(
        "من حالم خوب نیست",
        pol_T=[0.48, 0.52],
        generated="The person is smiling.",
        visual_hat="pos",
    )
    assert label == "positive_sarcasm"


def test_short_nonemotional_caption_is_not_sarcasm():
    label = assign_proposal_label(
        "عکس قدیمی #سلفی",
        pol_T=[0.90, 0.10],
        generated="The people are smiling.",
        visual_hat="pos",
        visual_pos=0.95,
    )
    assert label in {"positive", "neutral"}
    assert label != "positive_sarcasm"


def test_mourning_formula_plus_smile_is_positive_sarcasm():
    label = assign_proposal_label(
        "زنده‌یاد پدر، یادش گرامی",
        pol_T=[0.40, 0.60],
        generated="The person is smiling.",
        visual_hat="pos",
        visual_pos=0.9,
    )
    assert label == "positive_sarcasm"


def test_ad_copy_is_neutral():
    label = assign_proposal_label(
        "برای ثبت نام به واتساپ پیام دهید 02191691114 عرشیان",
        pol_T=[0.30, 0.70],
        pol_T_hat=[0.48, 0.52],
        generated="A professional in a suit.",
    )
    assert label == "neutral"
