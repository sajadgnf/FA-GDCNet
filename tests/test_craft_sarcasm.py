"""Crafted clash posts stay unlabeled until blind review."""

import json
from pathlib import Path

from data.craft_sarcasm import craft_from_existing, craft_pool
from data.enqueue_sarcasm import PLACEHOLDER_LABEL, enqueue
from data.tags import BOOTSTRAP_TAG, CRAFTED_TAG


def test_craft_pool_writes_caption_and_image(tmp_path: Path, monkeypatch) -> None:
    def fake_download(url: str, dest: Path, *, timeout: float = 30.0) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-jpg" * 8)

    monkeypatch.setattr("data.craft_sarcasm._download", fake_download)
    n = craft_pool(
        out_dir=tmp_path,
        pool_name="crafted",
        require_face=False,
        specs=(("craft-99", "http://example.com/a.jpg", "حالم بده ولی دارم میخندم"),),
    )
    assert n == 1
    rows = [
        json.loads(l)
        for l in (tmp_path / "crafted.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert rows[0]["post_id"] == "craft-99"
    assert (tmp_path / "images" / "craft-99.jpg").is_file()


def test_craft_from_existing_copies_image(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "face.jpg"
    src.write_bytes(b"fake-jpg")
    n = craft_from_existing(
        out_dir=tmp_path,
        pool_name="crafted",
        specs=(("craft-s99", str(src), "حالم بده ولی دارم میخندم"),),
    )
    assert n == 1
    dest = tmp_path / "images" / "craft-s99.jpg"
    assert dest.is_file()
    rows = [
        json.loads(l)
        for l in (tmp_path / "crafted.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert rows[0]["post_id"] == "craft-s99"


def test_enqueue_marks_crafted_as_bootstrap_not_gold(tmp_path: Path) -> None:
    img = tmp_path / "craft-01.jpg"
    img.write_bytes(b"x")
    gold = tmp_path / "gold.jsonl"
    gold.write_text("", encoding="utf-8")
    pool = tmp_path / "crafted.jsonl"
    pool.write_text(
        json.dumps(
            {
                "post_id": "craft-01",
                "caption": "حالم بده ولی دارم میخندم",
                "image_path": str(img),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    result = enqueue(
        dataset=gold,
        ids_path=tmp_path / "ids.txt",
        unfiltered_pools=[pool],
        filtered_pools=[],
        include_gold_heuristic=False,
        from_pools_only=True,
        roots=[tmp_path],
    )
    assert result.imported == 1
    row = json.loads(gold.read_text(encoding="utf-8").splitlines()[0])
    assert row["label"] == PLACEHOLDER_LABEL
    assert BOOTSTRAP_TAG in row["annotators"]
    assert CRAFTED_TAG in row["annotators"]


def test_craft_from_labeled_gold_pairs_valence(tmp_path: Path, monkeypatch) -> None:
    from data.craft_sarcasm import craft_from_labeled_gold

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    happy = img_dir / "happy.jpg"
    sad = img_dir / "sad.jpg"
    happy.write_bytes(b"h" * 64)
    sad.write_bytes(b"s" * 64)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "post_id": "g-happy",
                "caption": "امروز خوب بود",
                "image_path": str(happy),
                "label": "positive",
                "annotators": ["sjjd6502"],
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {
                "post_id": "g-sad",
                "caption": "حالم بد است",
                "image_path": str(sad),
                "label": "negative",
                "annotators": ["sjjd6502"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("data.face_filter.has_face", lambda *a, **k: True)
    n = craft_from_labeled_gold(
        dataset=gold,
        out_dir=tmp_path,
        pool_name="crafted",
        max_n=4,
        require_face=True,
        clip_gate=False,
    )
    from data.craft_sarcasm import (
        HAPPY_FACE_BITTER_CAPTIONS,
        SAD_FACE_CHEERFUL_CAPTIONS,
    )
    from data.preprocess import preprocess_caption

    assert n == 2
    rows = [
        json.loads(l)
        for l in (tmp_path / "crafted.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    by_id = {r["post_id"]: r for r in rows}
    assert "craft-hp000" in by_id
    assert "craft-sn000" in by_id
    assert by_id["craft-hp000"]["caption"] == preprocess_caption(HAPPY_FACE_BITTER_CAPTIONS[0])
    assert by_id["craft-sn000"]["caption"] == preprocess_caption(SAD_FACE_CHEERFUL_CAPTIONS[0])


def test_clip_gate_keeps_only_strong_matching_faces(tmp_path: Path, monkeypatch) -> None:
    from data.craft_sarcasm import craft_from_labeled_gold

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    smile = img_dir / "smile.jpg"
    frown = img_dir / "frown.jpg"
    weak = img_dir / "weak.jpg"
    smile.write_bytes(b"a" * 64)
    frown.write_bytes(b"b" * 64)
    weak.write_bytes(b"c" * 64)
    gold = tmp_path / "gold.jsonl"
    rows = [
        {
            "post_id": "g-smile",
            "caption": "خوبم",
            "image_path": str(smile),
            "label": "positive",
            "annotators": ["sjjd6502"],
        },
        {
            "post_id": "g-frown",
            "caption": "بدم",
            "image_path": str(frown),
            "label": "negative",
            "annotators": ["sjjd6502"],
        },
        {
            "post_id": "g-weak",
            "caption": "خوبم",
            "image_path": str(weak),
            "label": "positive",
            "annotators": ["sjjd6502"],
        },
    ]
    gold.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    affect = tmp_path / "affect.jsonl"
    affect.write_text(
        json.dumps({"post_id": "g-smile", "pos": 0.91, "neg": 0.09, "hat": "pos"})
        + "\n"
        + json.dumps({"post_id": "g-frown", "pos": 0.08, "neg": 0.92, "hat": "neg"})
        + "\n"
        + json.dumps({"post_id": "g-weak", "pos": 0.51, "neg": 0.49, "hat": "neu"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("data.face_filter.has_face", lambda *a, **k: True)
    n = craft_from_labeled_gold(
        dataset=gold,
        out_dir=tmp_path,
        pool_name="crafted",
        max_n=4,
        require_face=True,
        clip_gate=True,
        min_clip=0.70,
        affect_path=affect,
    )
    assert n == 2
    crafted = [
        json.loads(l)
        for l in (tmp_path / "crafted.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    ids = {r["post_id"] for r in crafted}
    assert ids == {"craft-vh000", "craft-vs000"}
    by = {r["post_id"]: r for r in crafted}
    assert by["craft-vh000"]["source_post_id"] == "g-smile"
    assert by["craft-vs000"]["source_post_id"] == "g-frown"


def test_clip_captions_are_unique():
    from data.craft_sarcasm import HAPPY_FACE_BITTER_CAPTIONS, SAD_FACE_CHEERFUL_CAPTIONS

    assert len(HAPPY_FACE_BITTER_CAPTIONS) == len(set(HAPPY_FACE_BITTER_CAPTIONS))
    assert len(SAD_FACE_CHEERFUL_CAPTIONS) == len(set(SAD_FACE_CHEERFUL_CAPTIONS))
    assert len(HAPPY_FACE_BITTER_CAPTIONS) >= 80
    assert len(SAD_FACE_CHEERFUL_CAPTIONS) >= 80
    meta = ("دوربین", "این خنده", "دروغ است", "ژست پیروزی", "برای استوری می‌خندم")
    props = (
        "گلدون",
        "بیسکویت",
        "گچ دست",
        "سنجاق",
        "تاول",
        "آژیر",
        "قابلمه",
        "نقاشی بچه",
        "چتر نداشتم",
        "رنگ مو ریخته",
        "کلیدو گم",
        "لبخند واقعی",
    )
    blob_h = " ".join(HAPPY_FACE_BITTER_CAPTIONS)
    blob_s = " ".join(SAD_FACE_CHEERFUL_CAPTIONS)
    for phrase in meta + props:
        assert phrase not in blob_h
        assert phrase not in blob_s


def test_recaption_skips_blind_and_rewrites_pending(tmp_path: Path) -> None:
    from data.craft_sarcasm import (
        HAPPY_FACE_BITTER_CAPTIONS,
        recaption_unlabeled_clip_queue,
    )
    from data.preprocess import preprocess_caption

    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "post_id": "craft-vh000",
                "caption": "قدیمی",
                "label": "positive_sarcasm",
                "annotators": ["blind-relabel"],
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {
                "post_id": "craft-vh001",
                "caption": "قدیمی",
                "label": "neutral",
                "annotators": ["weak-sarcasm-bootstrap"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    pool = tmp_path / "crafted.jsonl"
    pool.write_text("", encoding="utf-8")
    n = recaption_unlabeled_clip_queue(dataset=gold, pool=pool, skip_blind=True)
    assert n == 1
    rows = [json.loads(l) for l in gold.read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {r["post_id"]: r for r in rows}
    assert by["craft-vh000"]["caption"] == "قدیمی"
    assert by["craft-vh001"]["caption"] == preprocess_caption(HAPPY_FACE_BITTER_CAPTIONS[1])

