#!/usr/bin/env python3
"""Compare polarity scoring strategies on Persian captions.

Stage 1 (slow, needs weights) computes raw signals and caches them:

    python scripts/bench_polarity_heads.py --dump

Stage 2 (fast, pure numpy) evaluates combination rules on the cache:

    python scripts/bench_polarity_heads.py

Signals per caption:
- `snappfood`: ParsBERT sentiment head collapsed to (p_pos - p_neg)
- `proto_sims`: cosine similarity to every emotion prototype in mCLIP text
                space — semantic, so it needs no keyword list
- the same signals for a *de-negated* rewrite, so negation can be handled by
  flipping a semantic score instead of by special-casing phrases

TUNE cases drive rule selection; HELDOUT cases are only reported, so we can see
whether a rule generalises to wording it was never tuned on.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CACHE = ROOT / "artifacts" / "polarity_bench.json"

SNAPPFOOD_ID = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
CARDIFF_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# (caption, expected sign) — 0 means "should read as near-neutral".
TUNE_CASES: list[tuple[str, int]] = [
    ("ناراحتی", -1),
    ("ناراحتم", -1),
    ("ناراحت است", -1),
    ("او ناراحت است", -1),
    ("غم", -1),
    ("دلم شکست", -1),
    ("گریه‌ام گرفت", -1),
    ("خسته‌ام از همه چیز", -1),
    ("من خوردم زمین", -1),
    ("امروز تصادف کردم", -1),
    ("پول‌هام رو دزدیدند", -1),
    ("تنهام گذاشت و رفت", -1),
    ("اون اگه دوستم داشت که میموند", -1),
    ("کاش هیچ‌وقت نمی‌دیدمش", -1),
    ("شادروان از سختی دنیا رخت بست", -1),
    ("تسلیت می‌گم", -1),
    ("مرحوم پدرم را از دست دادم", -1),
    ("به درک", -1),
    ("گور پدرش", -1),
    ("حالم از این وضع بهم می‌خورد", -1),
    ("افتضاح بود، اصلا نپسندیدم", -1),
    ("خیلی بد شد، حالم گرفته شد", -1),
    ("خوشبخت", 1),
    ("خوشحالم", 1),
    ("چه روز زیبایی", 1),
    ("قبول شدم!", 1),
    ("بالاخره کارم درست شد", 1),
    ("عاشقتم", 1),
    ("دلم روشن شد", 1),
    ("بهترین روز عمرم بود", 1),
    ("لبخندش دنیا رو قشنگ می‌کنه", 1),
    ("سالگرد ازدواجمون مبارک", 1),
    ("عالی بود، ممنون از همه", 1),
    ("ناراحتی بود اما الان خوشحال است", 1),
    ("ناراحت نیستم", 1),
    ("امروز هوا ابری است", 0),
    ("ساعت سه جلسه داریم", 0),
    ("این کتاب دویست صفحه دارد", 0),
    ("فردا به تهران می‌روم", 0),
    ("قیمت دلار اعلام شد", 0),
    ("درک مطلب سخت است", -1),
]

HELDOUT_CASES: list[tuple[str, int]] = [
    ("امیدم رو از دست دادم", -1),
    ("دیشب سگم مرد", -1),
    ("خیلی داغونم", -1),
    ("چقدر بی‌رحمی", -1),
    ("همه چیز رو باختم", -1),
    ("مادرم بیمار شد", -1),
    ("خدا بیامرزدش", -1),
    ("برو بمیر", -1),
    ("خوشحال نیستم", -1),
    ("چه غذای خوشمزه‌ای", 1),
    ("ترفیع گرفتم", 1),
    ("خیلی ممنونم ازت", 1),
    ("حالم خیلی خوبه", 1),
    ("بچه‌ام به دنیا آمد", 1),
    ("دوستت دارم", 1),
    ("بد نبود", 1),
    ("مشکلی نیست", 1),
    ("این ساختمان ده طبقه است", 0),
    ("جلسه به هفته بعد موکول شد", 0),
    ("در حال مطالعه هستم", 0),
    # contrast clauses: the part after اما/ولی carries the claim
    ("روز بدی بود ولی شب خوبی داشتم", 1),
    ("اولش خوب بود اما بعد خراب شد", -1),
    ("خسته بودم اما خیلی خوشحالم", 1),
    ("همه چیز عالی بود ولی دلم شکست", -1),
]

_POS_LABELS = {"HAPPY", "POSITIVE", "POS", "LABEL_2", "DELIGHTED"}
_NEG_LABELS = {"SAD", "NEGATIVE", "NEG", "LABEL_0", "FURIOUS", "ANGRY"}

# Balanced concept regions: each prototype covers a semantic neighbourhood, so
# unseen wording in that neighbourhood still matches.
NEG_PROTOTYPES = [
    "متنی غمگین و ناراحت‌کننده",
    "حس بدی دارم و ناامید هستم",
    "خبر بد و اتفاق تلخ",
    "دلم شکست و گریه می‌کنم",
    "عصبانی و خسته و بیزارم",
    "درد و رنج و بیماری",
    "شکست خوردم و همه چیز خراب شد",
    "حالم گرفته شد و اوضاع ناجور است",
    "مرگ و از دست دادن عزیزان و سوگواری",
    "تسلیت و مجلس ترحیم و خاکسپاری",
    "بی‌تفاوتی تلخ و بی‌اهمیت شمردن",
    "توهین و فحش و بی‌احترامی",
]
POS_PROTOTYPES = [
    "متنی شاد و خوشحال‌کننده",
    "حس خوبی دارم و امیدوارم",
    "خبر خوب و اتفاق شیرین",
    "لبخند و خنده و شادی",
    "راضی و سپاسگزار و قدردان",
    "موفقیت و رسیدن به هدف",
    "عشق و محبت و مهربانی",
    "جشن و تولد و سالگرد مبارک",
    "زیبا و قشنگ و دلنشین",
    "آرامش و حال خوب",
    "سلامتی و تندرستی",
    "افتخار و پیروزی",
]
NEUTRAL_PROTOTYPES = [
    "یک جمله خبری معمولی و بی‌احساس",
    "توضیح ساده دربارهٔ زمان و مکان",
    "گزارش خنثی از یک واقعیت روزمره",
    "اطلاعات فنی و عددی بدون احساس",
    "برنامه و قرار ملاقات و ساعت جلسه",
    "توصیف عینی یک شیء یا مکان",
]

# Closed-class grammatical negation cues (function words, not sentiment words).
_NEGATION_RE = re.compile(
    r"(?:^|\s)(?:نیست\w*|نبود\w*|نباش\w*|نشد\w*|ندار\w*|نمی[\u200c\s]?\w+|هیچ|بدون|نه)(?=\s|$|[،,.!?؛;])"
)

# General de-negation rewrites: strip the negative prefix so the *same* semantic
# scorer can rate the underlying claim, then flip the sign.
_DENEG_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"نمی[\u200c\s]?"), "می‌"),
    (re.compile(r"\bنیستم\b"), "هستم"),
    (re.compile(r"\bنیستی\b"), "هستی"),
    (re.compile(r"\bنیستند\b"), "هستند"),
    (re.compile(r"\bنیست\b"), "است"),
    (re.compile(r"\bن(بود\w*|شد\w*|دار\w*|باش\w*|کرد\w*|رفت\w*)"), r"\1"),
    (re.compile(r"(?:^|\s)بدون\s"), " با "),
    (re.compile(r"(?:^|\s)هیچ\s"), " "),
]


_CONTRAST_RE = re.compile(r"(?:^|[\s،,؛;])(?:اما|ولی|لیکن)(?=[\s،,؛;]|$)")


def has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


def contrast_tail(text: str) -> str:
    """Clause after the last اما/ولی/لیکن, else the full text."""
    matches = list(_CONTRAST_RE.finditer(text))
    if not matches:
        return text
    return text[matches[-1].end() :].strip() or text


def denegate(text: str) -> str:
    out = text
    for pattern, repl in _DENEG_RULES:
        out = pattern.sub(repl, out)
    return re.sub(r"\s+", " ", out).strip()


def _all_cases() -> list[tuple[str, int, str]]:
    return [(t, w, "tune") for t, w in TUNE_CASES] + [
        (t, w, "heldout") for t, w in HELDOUT_CASES
    ]


# ---------------------------------------------------------------- stage 1 dump


def _head_scores(model_id: str, *text_lists: list[str]) -> list[list[float]]:
    """Signed (p_pos - p_neg) per text, for each supplied list of texts."""
    import torch
    from torch.nn.functional import softmax
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval().to(device)
    id2label = {int(i): str(n).upper() for i, n in model.config.id2label.items()}

    results = []
    for texts in text_lists:
        out = []
        for text in texts:
            inputs = tok(
                text, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)
            with torch.no_grad():
                probs = softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
            pos = sum(float(probs[i]) for i, n in id2label.items() if n in _POS_LABELS)
            neg = sum(float(probs[i]) for i, n in id2label.items() if n in _NEG_LABELS)
            out.append(pos - neg)
        results.append(out)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def _prototype_sims(texts: list[str]) -> dict:
    """Cosine similarity of each caption to every individual prototype."""
    from inference import models as M

    bundle = M.load_mclip_text_only()

    def unit(v: np.ndarray) -> np.ndarray:
        return v / max(float(np.linalg.norm(v)), 1e-8)

    groups = {"neg": NEG_PROTOTYPES, "pos": POS_PROTOTYPES, "neu": NEUTRAL_PROTOTYPES}
    embs = {
        k: np.stack([unit(M.embed_text_mclip(bundle, t)) for t in v]) for k, v in groups.items()
    }

    out: dict[str, list] = {"neg": [], "pos": [], "neu": []}
    for text in texts:
        v = unit(M.embed_text_mclip(bundle, text))
        for key, mat in embs.items():
            out[key].append([float(x) for x in (mat @ v)])
    M.release(bundle)
    return out


def dump() -> None:
    cases = _all_cases()
    texts = [t for t, _, _ in cases]
    deneg_texts = [denegate(t) for t in texts]

    tail_texts = [contrast_tail(t) for t in texts]
    tail_deneg_texts = [denegate(t) for t in tail_texts]

    (snap,) = _head_scores(SNAPPFOOD_ID, texts)
    cardiff, cardiff_deneg, cardiff_tail, cardiff_tail_deneg = _head_scores(
        CARDIFF_ID, texts, deneg_texts, tail_texts, tail_deneg_texts
    )

    payload = {
        "cases": [
            {"text": t, "want": w, "split": s, "deneg": d, "tail": tl}
            for (t, w, s), d, tl in zip(cases, deneg_texts, tail_texts, strict=True)
        ],
        "snappfood": snap,
        "cardiff": cardiff,
        "cardiff_deneg": cardiff_deneg,
        "cardiff_tail": cardiff_tail,
        "cardiff_tail_deneg": cardiff_tail_deneg,
        "proto_sims": _prototype_sims(texts),
        "proto_sims_deneg": _prototype_sims(deneg_texts),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {CACHE}")


# ------------------------------------------------------------ stage 2 evaluate


def _softmax_score(row: dict, temperature: float) -> float:
    s = np.array(
        [np.mean(row["neg"]), np.mean(row["pos"]), np.mean(row["neu"])], dtype=np.float64
    )
    p = np.exp((s - s.max()) / temperature)
    p /= p.sum()
    return float(p[1] - p[0])


def _margin_score(row: dict, scale: float) -> float:
    margin = float(np.mean(row["pos"]) - np.mean(row["neg"]))
    return float(np.tanh(margin / scale))


def _accuracy(scores: list[float], wants: list[int], *, deadzone: float = 0.3) -> tuple[int, list[bool]]:
    oks = []
    for score, want in zip(scores, wants, strict=True):
        if want == 0:
            oks.append(abs(score) < deadzone)
        else:
            oks.append(score * want > 0 and abs(score) >= 0.05)
    return sum(oks), oks


def evaluate() -> None:
    payload = json.loads(CACHE.read_text(encoding="utf-8"))
    cases = payload["cases"]
    texts = [c["text"] for c in cases]
    wants = [int(c["want"]) for c in cases]
    splits = [c["split"] for c in cases]
    snap = [float(x) for x in payload["snappfood"]]

    def rows(key: str) -> list[dict]:
        ps = payload[key]
        return [
            {"neg": ps["neg"][i], "pos": ps["pos"][i], "neu": ps["neu"][i]}
            for i in range(len(texts))
        ]

    plain = rows("proto_sims")
    tune = [i for i, s in enumerate(splits) if s == "tune"]
    held = [i for i, s in enumerate(splits) if s == "heldout"]

    def score_on(idx: list[int], scores: list[float]) -> int:
        hits, _ = _accuracy([scores[i] for i in idx], [wants[i] for i in idx])
        return hits

    print("zero-shot scoring form (tuned on TUNE split):")
    best_form, best_param, best_hits = "softmax", 0.02, -1
    for temp in (0.005, 0.01, 0.02, 0.03, 0.05):
        s = [_softmax_score(r, temp) for r in plain]
        hits = score_on(tune, s)
        print(f"  softmax T={temp:<6} {hits}/{len(tune)}")
        if hits > best_hits:
            best_form, best_param, best_hits = "softmax", temp, hits
    for scale in (0.005, 0.01, 0.02, 0.04, 0.08):
        s = [_margin_score(r, scale) for r in plain]
        hits = score_on(tune, s)
        print(f"  margin  s={scale:<6} {hits}/{len(tune)}")
        if hits > best_hits:
            best_form, best_param, best_hits = "margin", scale, hits
    print(f"  best: {best_form} param={best_param}\n")

    def zs(rows_: list[dict]) -> list[float]:
        if best_form == "softmax":
            return [_softmax_score(r, best_param) for r in rows_]
        return [_margin_score(r, best_param) for r in rows_]

    z_plain = zs(plain)
    car = [float(x) for x in payload["cardiff"]]
    car_deneg = [float(x) for x in payload["cardiff_deneg"]]
    is_neg = [has_negation(t) for t in texts]

    car_tail = [float(x) for x in payload["cardiff_tail"]]
    car_tail_deneg = [float(x) for x in payload["cardiff_tail_deneg"]]
    tails = [c["tail"] for c in cases]
    has_contrast = [t != full for t, full in zip(tails, texts, strict=True)]
    tail_is_neg = [has_negation(t) for t in tails]

    def cardiff_negation_aware(flip: float = 0.85) -> list[float]:
        """Negation: rate the de-negated claim, then flip its sign."""
        return [
            (-flip * d) if n else c
            for c, d, n in zip(car, car_deneg, is_neg, strict=True)
        ]

    def contrast_aware(tail_weight: float, *, flip: float = 0.85) -> list[float]:
        """On contrast, trust the final clause; negation still flips."""
        base = cardiff_negation_aware(flip)
        tail_scores = [
            (-flip * d) if n else c
            for c, d, n in zip(car_tail, car_tail_deneg, tail_is_neg, strict=True)
        ]
        return [
            (tail_weight * t + (1 - tail_weight) * b) if ctr else b
            for t, b, ctr in zip(tail_scores, base, has_contrast, strict=True)
        ]

    def with_semantic_prior(base: list[float], *, weak: float, weight: float) -> list[float]:
        """Let the semantic score decide when the head is unsure or clashes."""
        out = []
        for b, z in zip(base, z_plain, strict=True):
            unsure = abs(b) < weak
            clash = b * z < 0 and abs(z) >= 0.35
            out.append((1 - weight) * b + weight * z if (unsure or clash) else b)
        return out

    strategies: dict[str, list[float]] = {
        "snappfood only (current)": snap,
        "zero-shot only": z_plain,
        "cardiff only": car,
        "cardiff + negation flip": cardiff_negation_aware(),
        "cardiff + neg + prior(w=.6)": with_semantic_prior(
            cardiff_negation_aware(), weak=0.15, weight=0.6
        ),
        "cardiff + neg + prior(w=.8)": with_semantic_prior(
            cardiff_negation_aware(), weak=0.15, weight=0.8
        ),
        "cardiff + neg + prior(weak=.3,w=.6)": with_semantic_prior(
            cardiff_negation_aware(), weak=0.30, weight=0.6
        ),
        "cardiff + snappfood mean": [(a + c) / 2 for a, c in zip(snap, car, strict=True)],
        "cardiff + neg + contrast(.75)": contrast_aware(0.75),
        "cardiff + neg + contrast(1.0)": contrast_aware(1.0),
        "cardiff + neg + contrast(.75) + prior": with_semantic_prior(
            contrast_aware(0.75), weak=0.15, weight=0.6
        ),
    }

    print(f"{'strategy':<34}{'tune':>8}{'heldout':>10}{'total':>8}")
    results = {}
    for name, scores in strategies.items():
        t, h = score_on(tune, scores), score_on(held, scores)
        results[name] = (t + h, scores)
        print(f"{name:<34}{t:>4}/{len(tune):<3}{h:>6}/{len(held):<3}{t + h:>5}/{len(texts)}")

    winner = max(results, key=lambda k: results[k][0])
    print(f"\nbest strategy: {winner}\nper-case detail:")
    _, scores = results[winner]
    _, oks = _accuracy(scores, wants)
    for text, want, score, ok, split, n in zip(
        texts, wants, scores, oks, splits, is_neg, strict=True
    ):
        flag = " [neg]" if n else ""
        tag = "H" if split == "heldout" else " "
        print(f"  {tag} {'ok  ' if ok else 'MISS'} {score:+.3f}  want {want:+d}  {text}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true", help="recompute cached signals")
    args = ap.parse_args()
    if args.dump or not CACHE.is_file():
        dump()
    evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
