"""Build a small pool of face photos + Persian captions with a polarity clash.

Instagram hashtag pages almost never yield multimodal sarcasm (face vs caption).
These rows are still unlabeled: they go through the same blind 1–5 review.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import ssl
import urllib.request
from pathlib import Path

from .enqueue_sarcasm import _load_jsonl, _save_jsonl
from .eval_set import is_eval_eligible
from .preprocess import preprocess_caption
from .scrape import DEFAULT_RAW_DIR
from .tags import BLIND_REVIEW_TAG

log = logging.getLogger(__name__)

DEFAULT_POOL = Path("datasets") / "raw" / "crafted.jsonl"
DEFAULT_AFFECT = Path("artifacts") / "stages" / "image_affect.jsonl"
DEFAULT_MIN_CLIP = 0.70
SARCASM_LABELS = frozenset({"positive_sarcasm", "negative_sarcasm"})
_UA = "FA-GDCNet/1.0 (thesis dataset; educational use)"
_CTX = ssl.create_default_context()


def _spec_parts(spec: tuple) -> tuple[str, str, str, str]:
    post_id, src, caption = spec[0], spec[1], spec[2]
    source_id = str(spec[3]) if len(spec) > 3 else ""
    return post_id, src, caption, source_id


def _load_affect(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["post_id"])] = row
    return out


def _clip_pos_neg(affect: dict[str, dict], post_id: str) -> tuple[float, float] | None:
    row = affect.get(post_id)
    if not row or row.get("missing"):
        return None
    try:
        return float(row["pos"]), float(row["neg"])
    except (KeyError, TypeError, ValueError):
        return None


def _craft_image_sizes(image_dir: Path) -> set[int]:
    sizes: set[int] = set()
    if not image_dir.is_dir():
        return sizes
    for path in image_dir.glob("craft-*"):
        try:
            sizes.add(int(path.stat().st_size))
        except OSError:
            continue
    return sizes


def _existing_source_ids(jsonl_path: Path) -> set[str]:
    out: set[str] = set()
    for row in _load_jsonl(jsonl_path):
        sid = str(row.get("source_post_id") or "").strip()
        if sid:
            out.add(sid)
    return out


def _existing_post_ids(jsonl_path: Path) -> set[str]:
    return {str(row.get("post_id") or "") for row in _load_jsonl(jsonl_path) if row.get("post_id")}


def _next_index(existing_ids: set[str], prefix: str) -> tuple[int, int]:
    """Return (count, next_index) for ``prefix`` + zero-padded ints."""
    idxs: list[int] = []
    for pid in existing_ids:
        if not pid.startswith(prefix):
            continue
        tail = pid[len(prefix) :]
        if tail.isdigit():
            idxs.append(int(tail))
    return len(idxs), (max(idxs) + 1 if idxs else 0)


def _picsum(photo_id: int) -> str:
    return f"https://picsum.photos/id/{photo_id}/800/800"


# Lorem Picsum portrait ids (fallback when --from-existing is off).
CRAFT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("craft-01", _picsum(64), "حالم اصلاً خوب نیست؛ بهترین شب عمرمه. باور نکن"),
)

# Spoken Instagram captions that can sit under a *face* photo.
# Clash is face vs text. Do not name the camera, the smile, or «این خنده دروغ است».
# Do not caption objects that are not in the photo (biscuit, plaster, plant, …).
LOCAL_CRAFT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("craft-s01", "datasets/raw/images/DEkWqBPohL3.jpg", "عالی‌ام مرسی که پرسیدی 👍"),
    ("craft-s02", "datasets/raw/images/DDXaC53I6oP.jpg", "زندگی شیرینه، فقط قندش گرونه 😅"),
    ("craft-s03", "datasets/raw/images/DIyBdKtuIS0.jpg", "خوشبختم ها فقط اجاره رفته حقوق نیومده"),
    ("craft-s04", "datasets/raw/images/DDCe5TJMIgl.jpg", "بهترین سال عمرم بود حتما"),
    ("craft-s05", "datasets/raw/images/DWy4BqJDpBx.jpg", "شکر خدا، حسابم صفره"),
    ("craft-s06", "datasets/raw/images/DN3tuSoWJyc.jpg", "وضعیتمون فوق‌العاده‌ست باور نکن"),
    ("craft-n01", "datasets/raw/images/DIcJUuiByVt.jpg", "حال دلم خوبه نگران نباش"),
    ("craft-n02", "datasets/raw/images/DZX7CefCVQf.jpg", "دلتون شاد رفقا"),
    ("craft-n03", "datasets/raw/images/DZDSG58DfPG.jpg", "شکر، هیچی کم نداریم"),
    ("craft-n04", "datasets/raw/images/DCjBJz6s22t.jpg", "زندگی شیرینه دیگه"),
)

# Smile photos → bitter / fake-fine caption (intended 4). Unique; do not cycle.
HAPPY_FACE_BITTER_CAPTIONS: tuple[str, ...] = (
    "حال ما خوبه، تو باور نکن ☺️",
    "عالی‌ام مرسی که پرسیدی 👍",
    "زندگی شیرینه، فقط قندش گرونه 😅",
    "خوشبختم ها فقط اجاره رفته حقوق نیومده",
    "بهترین سال عمرم بود حتما",
    "شکر خدا، حسابم صفره",
    "وضعیتمون فوق‌العاده‌ست باور نکن",
    "قوی باش، بقیه‌ش مهم نیست 🙃",
    "خسته‌ام ولی خب می‌گذره دیگه",
    "امشب هم تنهام، عادت کردیم",
    "قول داده بود بمونه",
    "کسی حالمو نپرسید امروز",
    "حقوق اومد تا شبش تموم شد 😅",
    "زندگی لوکس با حداقل حقوق ✨",
    "سه شیفت کار می‌کنم بعد می‌گن خوش بحالت",
    "مادرم پرسید خوبی گفتم آره",
    "حالمو نپرس، همون همیشگیه",
    "می‌گذره، گفته بودن می‌گذره",
    "این هفته هم مثل هفته قبل",
    "دیگه انتظارم کم شده",
    "از این ماه سیر شدم",
    "رزومه فرستادم هنوز خبری نیست",
    "پیامشو خوندم جواب ندادم",
    "قرار کنسل شد، نرماله دیگه",
    "سرم شلوغه دلم خالی",
    "همه رفتن من موندم",
    "امروز حوصله حرف زدن ندارم",
    "کاش زودتر تموم شه امروز",
    "دلم برای خودم می‌سوزه",
    "گوشی بی‌صداست چون حوصله ندارم",
    "باز هم ماه تموم شد زود",
    "قبض زودتر از واریزی اومد",
    "عاشق این گرونی‌ام جدی",
    "حساب صفره باز دارم می‌چرخم",
    "شیفت تموم شد هنوز نخوابیدم",
    "خواب ندارم از دیروز",
    "سه شب بیدارم فردا شیفت دارم",
    "دعوا تموم شد بدون خداحافظی",
    "رفیق قدیمی رد شد سلام نکرد",
    "همکار ترفیع گرفت من تبریک گفتم",
    "کسی منتظرم نبود رسیدم خونه",
    "یه روز معمولی با قبض اضافه",
    "از بس صبر کردم عادت شد",
    "خبری نشد چون اصلا خبر نیومد",
    "پول کرایه موند حال هم موند",
    "دوست داشتم یکی باشه ببینه",
    "دارم ادامه می‌دم چون راه دیگه نیست",
    "شب‌ها شلوغ‌ترن از روزا",
    "از در که اومدم تو ساکت بود",
    "نامه اداره اومده بازش نکردم",
    "اینباکس پر تبلیغ خالی از آشنا",
    "نگران من نباشید خودم بلدم",
    "انرژی مثبت از قهوه نه از زندگی",
    "موفق شدم فقط نه تو اونی که می‌خواستم",
    "این ماه هم معجزه نشد",
    "آرزوهام کوچیک شدن اندازه حقوق",
    "فوق‌العاده‌ام تا وقتی حرف نزنم",
    "سالگرد تنهایی مبارک 🎉",
    "رابطه تموم شد ولی عالی‌ام",
    "خواب کافی؟ شوخی می‌کنی",
    "تعطیلات یعنی شیفت اضافه",
    "عاشق شغلم فقط حقوقش کمه 😅",
    "شکرگزارم بابت این همه قبض",
    "پلن A شکست، B هم همونه",
    "دوستام سفرن من شیفت‌ام",
    "تولد بود کسی نیومد مهم نیست",
    "وام تمدید شد جشن بگیریم",
    "دلار رفت بالا حال ما هم رفت بالا",
    "نون گرون شد ما هم راضی‌ایم",
    "دانشگاه تموم شد کار نیست عالیه",
    "همه ازدواج کردن من آزاد و بی‌پولم",
    "تا آخر ماه فقط قیمت‌ها رو نگاه می‌کنم",
    "بیمه رد شد درمان با خودمه",
    "بنزین گرون شد پیاده شادتریم",
    "چقدر آرومم فقط سه تا قهوه خوردم",
    "زندگیم مرتبه فقط خواب ندارم",
    "بهترین ورژن خودم بعد از اضافه‌کار",
    "حال خوب امروز موجود نیست",
    "تو باور نکن 🩶",
    "اگر حالمو می‌پرسی خدا رو شکر",
    "هیچکی نپرسید پس عالی بودم",
    "زندگیم در اوج، استطاعت نه",
)

# Sad/serious photos → fake-positive IG caption (intended 5). Unique; do not cycle.
SAD_FACE_CHEERFUL_CAPTIONS: tuple[str, ...] = (
    "امروز قشنگ گذشت ✨",
    "حال دلم خوبه نگران نباش",
    "دلتون شاد رفقا",
    "شکر، هیچی کم نداریم",
    "زندگی شیرینه دیگه",
    "عاشق این روزام",
    "آرامش یعنی این 🤍",
    "همه چی روبه‌راهه",
    "چه حال خوبی امروز",
    "از این بهتر نمیشه",
    "حقوق واریز شد حال کردیم",
    "خبر خوب رسید بالاخره",
    "بهترین ورژن خودمم امروز",
    "زندگیم عالیه جدی می‌گم",
    "انرژی‌م امروز خیلی بالاست",
    "حال دلم آرومه",
    "شکرگزار این روزام",
    "خوشبخت‌ترینم تو این لحظه",
    "همه چی داره درست می‌شه",
    "امروز یکی از بهترینه",
    "دلم قرصه نگران نیستم",
    "چقدر دنیای قشنگی داریم",
    "حس خوبی به این هفته دارم",
    "پرانرژی و آماده‌ام",
    "خوابم کامل بود سبک شدم",
    "کارا جمع شد زودتر تموم شد",
    "با مامان حرف زدم روزم ساخته شد",
    "شام با کسی بودم خوش گذشت",
    "امروز کسی منتظرم بود",
    "یه خبر کوچک ولی خیلی خوب",
    "هیچی معوق نمونده این ماه",
    "برنامه طبق وعده پیش رفت",
    "حس سبکی دارم نمی‌دونم چرا",
    "امروز دیر نرسیدم همه‌چی اوکی",
    "شب آروم تموم شد",
    "همه چی سر جاش بود",
    "پیام خوب آخر شب اومد",
    "از خواب که پا شدم سبک بودم",
    "یه چای درست حسابی حال داد",
    "به موقع رسیدم عجله نبود",
    "هوا خوبه حوصله دارم",
    "امروز انرژی داشتم",
    "خونه گرم بود آرامش داشتم",
    "تلفن که زنگ زد خوشحال شدم",
    "قرار سر جاش موند 🙏",
    "با همه تولد گرفتیم",
    "جواب‌ها خوب بودن بالاخره",
    "شیفت روز بود شب راحت خوابیدم",
    "با هم حرف زدیم درست شد",
    "کار رأس ساعت تموم شد",
    "مصاحبه آروم تموم شد",
    "وقت مصاحبه دادن خیالم راحت شد",
    "قبضو یکجا دادم خیالم راحت",
    "دوست قدیمی وایستاد حرف زدیم",
    "خواب خودبه‌خود اومد چه نعمتی",
    "یه کتاب خریدم بدون استرس",
    "از بیرون برگشتم حال داشتم",
    "یه حس خوب ته روز موند",
    "عاشق روتینم شدم",
    "امروز خودمو دوست داشتم",
    "ممنون بابت این آرامش",
    "زندگی داره به کامم می‌چرخه",
    "پر از امیدم این روزا",
    "حال خوش مال منه امروز",
    "دلم گرمه از این همه خوبی",
    "خیالم راحته دیگه",
    "هیچ نگرانی‌ای نمونده",
    "روزای قشنگ شروع شدن",
    "شکر این نعمت‌ها",
    "حالم بهتر از همیشه است",
    "انگار سنگ از رو دلم برداشته شد",
    "امروز فقط چیزای خوب دیدم",
    "این هفته بهم لطف کرد",
    "راضیم از خودم از زندگی",
    "قلبم سبکه امروز",
    "دنیا باهام مهربون‌تر شده",
    "حس موفقیت دارم کم‌کم",
    "همه‌چی طبق خواسته‌م پیش رفت",
    "از فردا هم نمی‌ترسم دیگه",
    "این همان حال خوبیه که می‌خواستم",
    "آرامش حاصل انتخاب‌های درست است",
    "خوشحال باش امروز مال توئه",
)


def _download(url: str, dest: Path, *, timeout: float = 30.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
        dest.write_bytes(resp.read())


def craft_pool(
    *,
    out_dir: Path = DEFAULT_RAW_DIR,
    pool_name: str = "crafted",
    require_face: bool = True,
    timeout: float = 30.0,
    specs: tuple[tuple[str, str, str], ...] = CRAFT_SPECS,
) -> int:
    image_dir = out_dir / "images"
    jsonl_path = out_dir / f"{pool_name}.jsonl"
    existing: set[str] = set()
    if jsonl_path.is_file():
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    existing.add(json.loads(line)["post_id"])

    written = 0
    skipped_no_face = 0
    with jsonl_path.open("a", encoding="utf-8") as out:
        for post_id, url, caption in specs:
            if post_id in existing:
                continue
            dest = image_dir / f"{post_id}.jpg"
            try:
                _download(url, dest, timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                log.warning("skip %s: download failed (%s)", post_id, exc)
                dest.unlink(missing_ok=True)
                continue
            if not dest.is_file() or dest.stat().st_size < 32:
                log.warning("skip %s: empty download", post_id)
                dest.unlink(missing_ok=True)
                continue
            if require_face:
                from .face_filter import has_face as _image_has_face

                if not _image_has_face(dest, min_size=32, min_rel_area=0.005):
                    skipped_no_face += 1
                    dest.unlink(missing_ok=True)
                    log.info("skip %s: no face in downloaded image", post_id)
                    continue
            row = {
                "post_id": post_id,
                "caption": preprocess_caption(caption),
                "image_path": str(dest),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing.add(post_id)
            written += 1
            log.info("crafted %s", post_id)
    if skipped_no_face:
        log.info("skipped %d downloads with no detected face", skipped_no_face)
    log.info("wrote %d crafted posts to %s", written, jsonl_path)
    return written


def craft_from_existing(
    *,
    out_dir: Path = DEFAULT_RAW_DIR,
    pool_name: str = "crafted",
    specs: tuple[tuple, ...] = LOCAL_CRAFT_SPECS,
    replace: bool = False,
) -> int:
    """Copy in-domain Instagram faces and pair them with clash captions."""
    image_dir = out_dir / "images"
    jsonl_path = out_dir / f"{pool_name}.jsonl"
    image_dir.mkdir(parents=True, exist_ok=True)
    if replace and jsonl_path.is_file():
        jsonl_path.unlink()
    existing: set[str] = set()
    if jsonl_path.is_file():
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    existing.add(json.loads(line)["post_id"])
    written = 0
    with jsonl_path.open("a", encoding="utf-8") as out:
        for spec in specs:
            post_id, src_s, caption, source_id = _spec_parts(spec)
            if post_id in existing:
                continue
            src = Path(src_s)
            if not src.is_file():
                log.warning("skip %s: missing source %s", post_id, src)
                continue
            dest = image_dir / f"{post_id}{src.suffix.lower()}"
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            row = {
                "post_id": post_id,
                "caption": preprocess_caption(caption),
                "image_path": str(dest),
            }
            if source_id:
                row["source_post_id"] = source_id
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing.add(post_id)
            written += 1
            log.info("crafted %s from %s", post_id, src.name)
    log.info("wrote %d recaptioned posts to %s", written, jsonl_path)
    return written


def _resolved_existing_sources(specs: tuple[tuple[str, str, str], ...]) -> set[str]:
    out: set[str] = set()
    for _pid, src, _cap in specs:
        path = Path(src)
        if path.is_file():
            out.add(str(path.resolve()))
    return out


def craft_from_labeled_gold(
    *,
    dataset: Path,
    out_dir: Path = DEFAULT_RAW_DIR,
    pool_name: str = "crafted",
    max_n: int = 120,
    require_face: bool = True,
    replace: bool = False,
    clip_gate: bool = True,
    min_clip: float = DEFAULT_MIN_CLIP,
    affect_path: Path = DEFAULT_AFFECT,
) -> int:
    """Recaption in-domain faces that CLIP already scores as smile vs frown.

    CLIP smile photos get a bitter caption (``craft-vh*``, intended 4).
    CLIP frown photos get a fake-cheerful caption (``craft-vs*``, intended 5).
    Gold ``positive``/``negative`` labels must match that face valence so we
    do not recaption a frowning 'positive' post. Original gold rows are not
    modified. New ids stay unlabeled until blind review.

    ``clip_gate=False`` falls back to gold labels alone (tests / old path).
    """
    from .face_filter import has_face as _image_has_face

    gold = _load_jsonl(dataset)
    jsonl_path = out_dir / f"{pool_name}.jsonl"
    used_src = _resolved_existing_sources(LOCAL_CRAFT_SPECS)
    used_ids = _existing_source_ids(jsonl_path)
    used_sizes = _craft_image_sizes(out_dir / "images")
    affect = _load_affect(affect_path) if clip_gate else {}
    if clip_gate and not affect:
        log.error("CLIP gate on but no scores at %s", affect_path)
        return 0

    happy_src: list[tuple[str, Path]] = []
    sad_src: list[tuple[str, Path]] = []
    for row in gold:
        pid = str(row.get("post_id") or "")
        if pid.startswith("craft-"):
            continue
        if pid in used_ids:
            continue
        if not is_eval_eligible(row.get("annotators")):
            continue
        label = str(row.get("label") or "")
        if label in SARCASM_LABELS:
            continue
        raw = str(row.get("image_path") or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_file():
            continue
        key = str(path.resolve())
        if key in used_src:
            continue
        try:
            size = int(path.stat().st_size)
        except OSError:
            continue
        if size in used_sizes:
            continue
        # CLIP smile/sad already implies a visible face. Haar misses many
        # frowning / small faces that CLIP still scores as sad.
        if require_face and not clip_gate:
            if not _image_has_face(path, min_size=48, min_rel_area=0.02):
                continue
        if clip_gate:
            scores = _clip_pos_neg(affect, pid)
            if scores is None:
                continue
            pos, neg = scores
            if label == "positive" and pos >= min_clip:
                used_src.add(key)
                used_ids.add(pid)
                happy_src.append((pid, path))
            elif label == "negative" and neg >= min_clip:
                used_src.add(key)
                used_ids.add(pid)
                sad_src.append((pid, path))
            continue
        if label not in {"positive", "negative"}:
            continue
        used_src.add(key)
        used_ids.add(pid)
        if label == "positive":
            happy_src.append((pid, path))
        else:
            sad_src.append((pid, path))

    half = max(0, int(max_n) // 2)
    specs: list[tuple] = []
    prefix_h, prefix_s = ("craft-vh", "craft-vs") if clip_gate else ("craft-hp", "craft-sn")
    existing_pids = _existing_post_ids(jsonl_path)
    have_h, next_h = _next_index(existing_pids, prefix_h)
    have_s, next_s = _next_index(existing_pids, prefix_s)
    need_h = max(0, half - have_h)
    need_s = max(0, half - have_s)
    for i, (src_id, path) in enumerate(happy_src[:need_h]):
        idx = next_h + i
        if idx >= len(HAPPY_FACE_BITTER_CAPTIONS):
            log.warning("ran out of unique bitter captions at index %d", idx)
            break
        cap = HAPPY_FACE_BITTER_CAPTIONS[idx]
        specs.append((f"{prefix_h}{idx:03d}", str(path), cap, src_id))
    for i, (src_id, path) in enumerate(sad_src[:need_s]):
        idx = next_s + i
        if idx >= len(SAD_FACE_CHEERFUL_CAPTIONS):
            log.warning("ran out of unique cheerful captions at index %d", idx)
            break
        cap = SAD_FACE_CHEERFUL_CAPTIONS[idx]
        specs.append((f"{prefix_s}{idx:03d}", str(path), cap, src_id))
    log.info(
        "labeled-face sources: %d smile, %d frown -> %d crafted specs "
        "(clip_gate=%s min=%.2f have=%d/%d need=%d/%d)",
        len(happy_src[:need_h]),
        len(sad_src[:need_s]),
        len(specs),
        clip_gate,
        min_clip,
        have_h,
        have_s,
        need_h,
        need_s,
    )
    return craft_from_existing(
        out_dir=out_dir,
        pool_name=pool_name,
        specs=tuple(specs),
        replace=replace,
    )


def recaption_unlabeled_clip_queue(
    *,
    dataset: Path,
    pool: Path = DEFAULT_POOL,
    skip_blind: bool = True,
) -> int:
    """Rewrite crafted captions to selfie-native Instagram Persian.

    Prefixes: ``craft-vh``/``craft-hp``/``craft-s`` bitter; ``craft-vs``/``craft-sn``/``craft-n`` cheerful.
    ``skip_blind=True`` leaves already-reviewed gold rows alone.
    """

    def _new_caption(pid: str) -> str | None:
        rules = (
            ("craft-vh", HAPPY_FACE_BITTER_CAPTIONS),
            ("craft-vs", SAD_FACE_CHEERFUL_CAPTIONS),
            ("craft-hp", HAPPY_FACE_BITTER_CAPTIONS),
            ("craft-sn", SAD_FACE_CHEERFUL_CAPTIONS),
            ("craft-s", HAPPY_FACE_BITTER_CAPTIONS),
            ("craft-n", SAD_FACE_CHEERFUL_CAPTIONS),
        )
        for prefix, bank in rules:
            if not pid.startswith(prefix):
                continue
            tail = pid[len(prefix) :]
            if not tail.isdigit():
                return None
            idx = int(tail)
            if idx >= len(bank):
                log.warning("no unique caption for %s (need index %d)", pid, idx)
                return None
            return preprocess_caption(bank[idx])
        return None

    changed = 0
    gold = _load_jsonl(dataset)
    by_id: dict[str, str] = {}
    for row in gold:
        pid = str(row.get("post_id") or "")
        tags = [str(a) for a in (row.get("annotators") or [])]
        if skip_blind and BLIND_REVIEW_TAG in tags:
            continue
        new = _new_caption(pid)
        if new is None or row.get("caption") == new:
            continue
        row["caption"] = new
        by_id[pid] = new
        changed += 1
    if changed:
        _save_jsonl(dataset, gold)
    pool_rows = _load_jsonl(pool)
    pool_changed = False
    for row in pool_rows:
        pid = str(row.get("post_id") or "")
        if pid in by_id:
            row["caption"] = by_id[pid]
            pool_changed = True
        elif pid not in {str(r.get("post_id")) for r in gold}:
            new = _new_caption(pid)
            if new and row.get("caption") != new:
                row["caption"] = new
                pool_changed = True
    if pool_changed:
        _save_jsonl(pool, pool_rows)
    log.info("recaptioned %d crafted captions", changed)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pair face photos with clash captions (still unlabeled)."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--pool-name", default="crafted")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--no-require-face", action="store_true")
    parser.add_argument(
        "--from-existing",
        action="store_true",
        help="Recaption Instagram faces already on disk (no download).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite the crafted pool JSONL before writing.",
    )
    parser.add_argument(
        "--from-labeled",
        action="store_true",
        help="Recaption faces from labeled positive/negative gold (clash captions).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets") / "persian_multimodal_irony.jsonl",
    )
    parser.add_argument("--max", type=int, default=120, dest="max_n")
    parser.add_argument(
        "--no-clip-gate",
        action="store_true",
        help="Use gold positive/negative labels as face valence (ignore CLIP).",
    )
    parser.add_argument(
        "--min-clip",
        type=float,
        default=DEFAULT_MIN_CLIP,
        help="Minimum CLIP smile/frown probability to recaption a face.",
    )
    parser.add_argument(
        "--affect",
        type=Path,
        default=DEFAULT_AFFECT,
        help="image_affect.jsonl with CLIP pos/neg scores.",
    )
    parser.add_argument(
        "--recaption-queue",
        action="store_true",
        help="Rewrite crafted captions in gold (also labeled rows unless --pending-only).",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="With --recaption-queue, skip rows that already have blind-relabel.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.recaption_queue:
        n = recaption_unlabeled_clip_queue(
            dataset=args.dataset,
            pool=args.out_dir / f"{args.pool_name}.jsonl",
            skip_blind=args.pending_only,
        )
        print(f"recaptioned={n}")
        return 0
    if args.from_labeled:
        n = craft_from_labeled_gold(
            dataset=args.dataset,
            out_dir=args.out_dir,
            pool_name=args.pool_name,
            max_n=args.max_n,
            require_face=not args.no_require_face,
            replace=args.replace,
            clip_gate=not args.no_clip_gate,
            min_clip=args.min_clip,
            affect_path=args.affect,
        )
    elif args.from_existing:
        n = craft_from_existing(
            out_dir=args.out_dir,
            pool_name=args.pool_name,
            replace=args.replace,
        )
    else:
        n = craft_pool(
            out_dir=args.out_dir,
            pool_name=args.pool_name,
            require_face=not args.no_require_face,
            timeout=args.timeout,
        )
    print(f"crafted={n} pool={args.out_dir / f'{args.pool_name}.jsonl'}")
    print("These are not gold. Enqueue then label:")
    print(
        "  python tasks.py enqueue-sarcasm --from-pools-only "
        f"--unfiltered-pool {args.out_dir / f'{args.pool_name}.jsonl'} "
        "--ids-file datasets/sarcasm_face_ids.txt --no-gold-heuristic"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
