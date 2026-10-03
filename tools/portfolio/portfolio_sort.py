#!/usr/bin/env python3
"""Разбор фото портфолио: инвентаризация, дубли, категории, имена, лёгкие копии, отчёт.

    python portfolio_sort.py run       # analyze + apply
    python portfolio_sort.py analyze   # только анализ -> _служебное/разметка.csv
    python portfolio_sort.py apply     # раскладка по разметке (её можно поправить руками)
    python portfolio_sort.py undo      # вернуть дубли из _дубли на исходные места

Оригиналы не меняются. Единственное действие с ними — дубли ПЕРЕНОСЯТСЯ (не удаляются)
в «<папка с фото>/_дубли». Всё остальное — копии.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import io
import json
import os
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageCms, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # без HEIC-поддержки .heic попадут в «битые»
    pillow_heif = None

Image.MAX_IMAGE_PIXELS = 400_000_000  # панорамы — не «декомпрессионная бомба»

if hasattr(sys.stdout, "reconfigure"):  # консоль Windows не всегда UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- настройки

DEFAULT_SRC = Path(r"C:\Users\Sasha\Jarvis\фото мебель")
SERVICE, DUPS, UNSORTED = "_служебное", "_дубли", "_разобрать"
SKIP_DIRS = {SERVICE, DUPS, UNSORTED}

IMAGE_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".heic", ".heif",
             ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".3gp", ".mkv", ".m4v", ".wmv", ".webm"}
SYSTEM_NAMES = {"thumbs.db", "desktop.ini", ".ds_store"}
KIND_RU = {"video": "видео", "other": "не изображение", "system": "системный файл"}

MAX_SIDE = 2000        # длинная сторона копий в Портфолио
JPEG_QUALITY = 85
ANALYSIS_SIDE = 1024   # размер, на котором считаются ORB, резкость, яркость

MODEL = ("ViT-L-14", "datacomp_xl_s13b_b90k")
CACHE_VERSION = 2

# Дубли
CAND_CLIP = 0.90       # кандидаты: похожесть по смыслу (CLIP, косинус) ...
CAND_PHASH = 12        # ... или близкий perceptual hash (из 64 бит)
SURE_PHASH = 4         # phash настолько близок, что ORB-проверка не нужна (при CLIP >= 0.85)
SAME_NCC = 0.5        # корреляция мелких деталей после совмещения кадров
SIMILAR_REPORT = 0.95  # «очень похожи, но разные кадры» — показать в проверке

# Категории
MIN_CONF = 0.45        # ниже — в _разобрать
MIN_MARGIN = 0.12      # отрыв от второй категории

# Проблемы
DARK_LUM = 55          # средняя яркость 0..255
BLUR_VAR = 60.0        # дисперсия лапласиана самого резкого участка кадра (на 1024 px)
SMALL_SIDE = 1000      # длинная сторона меньше — «низкое разрешение»

# ---------------------------------------------------------------- категории

# key: (папка, slug для имени, род для цвета, подписи для CLIP)
CATEGORIES = {
    "kitchen": ("Кухни", "kuhnya", "f", [
        "a photo of a kitchen with cabinets and a countertop",
        "a modern fitted kitchen with a sink, a hob and a range hood",
        "kitchen cabinets with built-in oven and fridge",
        "a corner kitchen in an apartment",
        "kitchen furniture, upper and lower cabinets",
    ]),
    "wardrobe": ("Шкафы", "shkaf", "m", [
        "a photo of a built-in wardrobe with sliding doors",
        "a tall wardrobe closet with hinged doors in a bedroom",
        "a floor-to-ceiling cabinet with doors",
        "a wardrobe with mirrored sliding doors",
        "a built-in bookcase and shelving unit",
    ]),
    "hallway": ("Прихожие", "prihozhaya", "f", [
        "a photo of an entrance hallway with a coat rack and a mirror",
        "hallway furniture with a shoe cabinet, coat hooks and a bench",
        "an apartment entryway next to the front door with a built-in closet",
        "a narrow corridor with a wardrobe and a mirror",
    ]),
    "bathroom": ("Ванные", "vannaya", "f", [
        "a photo of a bathroom vanity cabinet with a washbasin",
        "a bathroom with a sink cabinet, a mirror and tiles",
        "a wall-hung vanity unit under a sink in a bathroom",
        "a bathroom with a toilet, a shower and storage cabinets",
    ]),
    "walkin": ("Гардеробные", "garderobnaya", "f", [
        "a photo of a walk-in closet with open shelves and hanging rails",
        "a dressing room with clothes on hangers and shelves with shoes",
        "a walk-in wardrobe room with a storage system",
        "an open wardrobe system with clothes and no doors",
    ]),
    "tumba": ("Тумбы", "tumba", "f", [
        "a photo of a TV stand",
        "a low chest of drawers",
        "a bedside nightstand with drawers",
        "a low sideboard cabinet standing against a wall",
        "a wall-mounted floating TV unit in a living room",
    ]),
    "project": ("Проекты квартир", "proekt", "m", [
        "a floor plan of an apartment",
        "an open-plan apartment with a kitchen area and a living room in one space",
        "a 3D visualization rendering of an apartment interior design",
        "a collage of several photos of different rooms",
        "a studio apartment with a kitchen and a living area",
    ]),
}
FOLDERS = {key: c[0] for key, c in CATEGORIES.items()}
FOLDER_TO_KEY = {v.lower(): k for k, v in FOLDERS.items()}

# Не мебель Максима — в _разобрать с причиной
OTHER = {
    "скриншот": ["a screenshot of a phone screen", "a screenshot of a messenger chat",
                 "a screenshot of a website"],
    "документ/текст": ["a document with text", "a scanned paper document", "a price list or an invoice"],
    "чертёж/эскиз": ["a technical drawing with dimensions", "a hand-drawn sketch of furniture",
                     "a CAD drawing of a cabinet"],
    "деталь крупным планом": ["a close-up of a furniture hinge", "a close-up of a cabinet edge and a countertop corner",
                              "a macro photo of a drawer handle", "a close-up detail of furniture hardware"],
    "люди": ["a selfie of a person", "a portrait of a person", "a group of people"],
    "мастерская/материалы": ["a carpentry workshop with machines", "stacks of chipboard panels and wooden boards",
                             "furniture parts and boards before assembly", "screws and tools on a workbench"],
    "другая мебель": ["a bed in a bedroom", "a sofa in a living room", "a desk with a chair",
                      "a dining table with chairs"],
    "здание/улица": ["a photo of a building facade from the street", "the exterior of a house",
                     "a street with cars and buildings"],
    "не по теме": ["a landscape with trees", "a garden", "a cat or a dog", "food on a plate", "a car"],
    "неразборчиво": ["a very dark blurry photo", "a black image", "a blurry photo of nothing"],
}

# Цвет фасадов: (ж.р., м.р.), слово для подписи, доп. подписи. Один цвет может стоять в нескольких строках —
# их вероятности складываются. «Белые шкафы с чёрной столешницей» отдельно, иначе CLIP называет такую кухню чёрной.
COLORS = [
    (("belaya", "belyy"), "white", []),
    (("belaya", "belyy"), None, ["white cabinets with a dark black countertop", "white cabinet doors with black handles"]),
    (("seraya", "seryy"), "grey", []),
    (("chernaya", "chernyy"), "black", ["matte black cabinets"]),
    (("bezhevaya", "bezhevyy"), "beige", []),
    (("pod-derevo", "pod-derevo"), "natural wood", ["wooden {noun} {parts}"]),
    (("zelenaya", "zelenyy"), "green", []),
    (("sinyaya", "siniy"), "dark blue", []),
    (("dvuhtsvetnaya", "dvuhtsvetnyy"), None, ["a two-tone {noun} with {parts} in two contrasting colors",
                                               "upper and lower cabinets in different colors"]),
]
COLOR_NOUN = {"kitchen": ("kitchen", "cabinets"), "wardrobe": ("wardrobe", "doors"),
              "hallway": ("hallway furniture", "doors"), "bathroom": ("bathroom vanity", "drawers"),
              "walkin": ("walk-in closet", "shelves"), "tumba": ("cabinet", "drawers")}


def color_prompts(word, extra, noun, parts) -> list[str]:
    base = [f"a {noun} with {word} cabinet doors", f"{word} {noun} {parts}"] if word else []
    return base + [x.format(noun=noun, parts=parts) for x in extra]


TYPES = {
    "kitchen": {"pryamaya": ["a straight single-wall kitchen along one wall"],
                "uglovaya": ["an L-shaped corner kitchen"],
                "p-obraznaya": ["a U-shaped kitchen along three walls"],
                "s-ostrovom": ["a kitchen with a kitchen island in the middle"]},
    "wardrobe": {"kupe": ["a wardrobe with sliding doors"],
                 "raspashnoy": ["a wardrobe with hinged swing doors and handles"],
                 "s-zerkalom": ["a wardrobe with mirror doors"],
                 "napolnenie": ["an open wardrobe showing shelves and rails inside"]},
    "hallway": {"s-zerkalom": ["hallway furniture with a large mirror"],
                "s-veshalkoy": ["hallway furniture with open coat hooks"],
                "so-shkafom": ["a hallway with a tall closed closet"]},
    "bathroom": {"podvesnaya": ["a floating wall-hung bathroom vanity"],
                 "napolnaya": ["a floor-standing bathroom vanity cabinet on legs"],
                 "s-penalom": ["a bathroom with a tall narrow cabinet column"]},
    "walkin": {"otkrytaya": ["an open walk-in closet with shelves and rails, without doors"],
               "s-dveryami": ["a walk-in closet with cabinet doors"],
               "uglovaya": ["an L-shaped corner walk-in closet"]},
    "tumba": {"pod-tv": ["a TV stand"],
              "prikrovatnaya": ["a bedside nightstand"],
              "komod": ["a chest of drawers"],
              "podvesnaya": ["a wall-mounted floating cabinet"]},
    "project": {"plan": ["a floor plan of an apartment"],
                "vizualizatsiya": ["a 3D rendering of an interior design"],
                "kuhnya-gostinaya": ["an open-plan kitchen and living room"],
                "kollazh": ["a collage of several photos"]},
}
FINISH = {"glyanets": ["glossy high-gloss reflective cabinet fronts"],
          "": ["matte cabinet fronts", "plain cabinet fronts"],
          "shpon": ["natural wood veneer cabinet fronts"]}

# ---------------------------------------------------------------- мелочи


def log(msg: str = "") -> None:
    print(msg, flush=True)


def human(n: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.0f} {unit}" if unit in ("Б", "КБ") else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


TRANSLIT = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p",
                     "r", "s", "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))


def slugify(text: str) -> str:
    """Только латиница, цифры и дефисы — кириллицу транслитерируем."""
    text = "".join(TRANSLIT.get(ch, ch) for ch in (text or "").strip().lower())
    text = "".join(ch if ch.isascii() and ch.isalnum() else "-" for ch in text)
    return "-".join(filter(None, text.split("-")))


def popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 10_000):
        cand = path.with_name(f"{path.stem} ({i}){path.suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"не подобрать свободное имя для {path}")


def to_rgb(im: Image.Image) -> Image.Image:
    """В sRGB без альфы: прозрачность — на белый фон, цветовой профиль — в sRGB."""
    icc = im.info.get("icc_profile")
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    if icc and im.mode in ("RGB", "CMYK"):
        try:
            prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            if "srgb" not in (ImageCms.getProfileDescription(prof) or "").lower():
                im = ImageCms.profileToProfile(im, prof, ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception:
            pass  # битый или чужой профиль — оставляем как есть
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def open_image(path: Path, draft: int | None = None) -> Image.Image:
    im = Image.open(path)
    if draft and im.format == "JPEG":
        im.draft("RGB", (draft, draft))  # быстрое декодирование в уменьшенном масштабе
    im.load()
    im = ImageOps.exif_transpose(im)
    return to_rgb(im)


# ---------------------------------------------------------------- инвентаризация


def read_journal(svc: Path) -> dict[str, str]:
    """Исходный путь -> где файл лежит сейчас (для дублей, перенесённых прошлыми запусками)."""
    jp = svc / "перемещения.jsonl"
    moved = {}
    if jp.exists():
        for line in jp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line)
                moved[j["from"]] = j["to"]
    return moved


def locate(src: Path, rel: str, moved: dict[str, str]) -> Path:
    p = src / rel
    if not p.exists() and rel in moved and (src / moved[rel]).exists():
        return src / moved[rel]
    return p


def scan(src: Path, moved: dict[str, str] | None = None) -> list[dict]:
    """Все файлы, кроме служебных папок. Дубли, перенесённые прошлым запуском, учитываются на старом месте,
    чтобы повторный анализ давал тот же результат."""
    files = []
    for root, dirs, names in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            p = Path(root) / name
            ext = p.suffix.lower()
            if name.lower() in SYSTEM_NAMES or name.startswith("._"):
                kind = "system"
            elif ext in IMAGE_EXT:
                kind = "image"
            elif ext in VIDEO_EXT:
                kind = "video"
            else:
                kind = "other"
            st = p.stat()
            files.append({"rel": p.relative_to(src).as_posix(), "size": st.st_size,
                          "mtime": st.st_mtime, "ext": ext or "(без расширения)", "kind": kind})
    seen = {f["rel"] for f in files}
    for rel, now in (moved or {}).items():
        p = src / now
        if rel not in seen and p.is_file():
            st = p.stat()
            files.append({"rel": rel, "path": now, "size": st.st_size, "mtime": st.st_mtime,
                          "ext": p.suffix.lower() or "(без расширения)", "kind": "image"})
    files.sort(key=lambda f: f["rel"])
    return files


def inventory_summary(files: list[dict]) -> dict:
    by_ext = defaultdict(lambda: [0, 0])
    for f in files:
        by_ext[f["ext"]][0] += 1
        by_ext[f["ext"]][1] += f["size"]
    kinds = Counter(f["kind"] for f in files)
    return {"total": len(files), "bytes": sum(f["size"] for f in files),
            "by_ext": dict(sorted(by_ext.items(), key=lambda kv: -kv[1][0])),
            "kinds": dict(kinds)}


def print_inventory(inv: dict) -> None:
    log(f"Всего файлов: {inv['total']}, {human(inv['bytes'])}")
    k = inv["kinds"]
    log(f"  фото: {k.get('image', 0)}, видео: {k.get('video', 0)}, "
        f"прочее: {k.get('other', 0)}, системные: {k.get('system', 0)}")
    for ext, (n, size) in inv["by_ext"].items():
        log(f"  {ext:18s} {n:5d}  {human(size)}")


# ---------------------------------------------------------------- признаки


class Clip:
    def __init__(self, cache_dir: Path, model=MODEL):
        os.environ.setdefault("HF_HOME", str(cache_dir))  # модель — внутри _служебное
        import open_clip
        import torch

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.torch = torch
        log(f"Загрузка модели {model[0]} ({model[1]}) — первый раз скачивается ~1.7 ГБ…")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model[0], pretrained=model[1], cache_dir=str(cache_dir))
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model[0])

    def encode_images(self, tensors: list) -> np.ndarray:
        with self.torch.no_grad():
            emb = self.model.encode_image(self.torch.stack(tensors))
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().numpy()

    def encode_texts(self, prompts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            emb = self.model.encode_text(self.tokenizer(prompts))
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().numpy()

    def class_matrix(self, groups: list[list[str]]) -> np.ndarray:
        """Одна строка на класс: среднее нормированных эмбеддингов его подписей."""
        rows = []
        for prompts in groups:
            e = self.encode_texts(prompts).mean(0)
            rows.append(e / np.linalg.norm(e))
        return np.stack(rows)


def square_pad(im: Image.Image) -> Image.Image:
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas


def sharpness(gray: np.ndarray) -> float:
    """Резкость самого резкого участка (2-й из 16), чтобы фото с размытым фоном не считались размытыми."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = lap.shape
    if min(h, w) < 32:
        return float(lap.var())
    tiles = [lap[y * h // 4:(y + 1) * h // 4, x * w // 4:(x + 1) * w // 4].var() for y in range(4) for x in range(4)]
    return float(sorted(tiles)[-2])


def broken_reason(e: Exception) -> str:
    text = str(e)
    if "truncated" in text:
        return "файл обрезан (не докачан/не дописан)"
    if type(e).__name__ == "UnidentifiedImageError":
        return "не распознаётся как изображение (повреждён или другой формат)"
    if isinstance(e, Image.DecompressionBombError):
        return "слишком большое изображение"
    return f"не открывается: {type(e).__name__}: {text}"[:200]


def safe_extract(src: Path, rec: dict, preprocess) -> dict:
    """Ошибка на одном файле не должна останавливать разбор полутора тысяч."""
    try:
        return extract(src, rec, preprocess)
    except Exception as e:
        out = dict(rec)
        out["broken"] = f"ошибка анализа: {type(e).__name__}: {e}"[:200]
        return out


def extract(src: Path, rec: dict, preprocess) -> dict:
    """Всё, что нужно знать о файле: хеши, размеры, яркость, резкость, ORB, тензор для CLIP."""
    path = src / rec.get("path", rec["rel"])
    out = dict(rec)
    try:
        data = path.read_bytes()
        out["sha256"] = hashlib.sha256(data).hexdigest()
        im = Image.open(io.BytesIO(data))
        out["format"] = im.format
        w, h = im.size
        exif = im.getexif()
        if exif.get(0x0112, 1) in (5, 6, 7, 8):
            w, h = h, w
        out["width"], out["height"] = w, h
        out["gps"] = bool(exif.get_ifd(0x8825))
        out["taken"] = str(exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132) or "")
        if im.format == "JPEG":
            im.draft("RGB", (ANALYSIS_SIDE, ANALYSIS_SIDE))
        im.load()
        im = to_rgb(ImageOps.exif_transpose(im))
    except Exception as e:  # битый, обрезанный, неизвестный формат
        out["broken"] = broken_reason(e)
        return out

    im.thumbnail((ANALYSIS_SIDE, ANALYSIS_SIDE), Image.LANCZOS)
    gray = np.asarray(im.convert("L"))
    out["lum"] = float(gray.mean())
    out["sharp"] = sharpness(gray)
    g = im.copy()
    g.thumbnail((256, 256), Image.LANCZOS)
    out["g256"] = np.asarray(g.convert("L"))

    small = im.copy()
    small.thumbnail((256, 256), Image.LANCZOS)
    out["phash"] = [int(str(imagehash.phash(small.rotate(r, expand=True))), 16) for r in (0, 90, 180, 270)]

    orb = cv2.ORB_create(nfeatures=1500)
    kps, desc = orb.detectAndCompute(gray, None)
    out["kp"] = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
    out["desc"] = desc
    out["thumb_wh"] = im.size
    # центральный квадрат + вся картинка с полями: интерьер часто шире квадрата
    out["_tensors"] = [preprocess(im), preprocess(square_pad(im))]
    return out


# ---------------------------------------------------------------- дубли


def _highpass(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 3.0)


def same_frame(a: dict, b: dict) -> tuple[bool, str]:
    """Один и тот же кадр (кроп, пережатие, поворот, уменьшение)?

    1) ORB-точки совпадают через поворот+масштаб+сдвиг (RANSAC);
    2) один кадр почти целиком лежит внутри другого;
    3) после совмещения мелкая структура картинок совпадает (корреляция высоких частот).
    Разные ракурсы этой проверки не проходят: из-за параллакса совмещение не сходится.
    """
    da, db = a.get("desc"), b.get("desc")
    if da is None or db is None or len(da) < 15 or len(db) < 15:
        return False, "мало точек"
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 12:
        return False, f"совпадений {len(good)}"
    pa = a["kp"][[m.queryIdx for m in good]]
    pb = b["kp"][[m.trainIdx for m in good]]
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                         maxIters=3000, confidence=0.995)
    if M is None:
        return False, "нет преобразования"
    n_in, ratio = int(inl.sum()), float(inl.mean())
    if n_in < 10 or ratio < 0.4:
        return False, f"точек {n_in}/{len(good)}"

    ga, gb = a["g256"], b["g256"]
    sa, sb = ga.shape[1] / a["thumb_wh"][0], gb.shape[1] / b["thumb_wh"][0]
    m = np.diag([sb, sb, 1.0]) @ np.vstack([M, [0, 0, 1]]) @ np.diag([1 / sa, 1 / sa, 1.0])
    h, w = gb.shape
    warped = cv2.warpAffine(ga.astype(np.float32), m[:2], (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.ones(ga.shape, np.uint8), m[:2], (w, h), flags=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
    if mask.sum() < 500:
        return False, "нет перекрытия"
    cover_b = float(mask.mean())
    cover_a = float(min(1.0, mask.sum() / (abs(np.linalg.det(m[:2, :2])) * ga.size)))
    x, y = _highpass(warped)[mask], _highpass(gb)[mask]
    x, y = x - x.mean(), y - y.mean()
    ncc = float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-6))
    ok = max(cover_a, cover_b) >= 0.8 and min(cover_a, cover_b) >= 0.25 and ncc >= SAME_NCC
    return ok, f"точек {n_in}/{len(good)}, перекрытие {min(cover_a, cover_b):.0%}, сходство {ncc:.2f}"


def find_duplicates(items: list[dict]) -> tuple[list[list[int]], dict, list[tuple]]:
    """items — только читаемые фото. Возвращает группы индексов, причины и пары «похожи, но разные»."""
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    reason = {}
    # 1. точные копии
    by_sha = defaultdict(list)
    for i, it in enumerate(items):
        by_sha[it["sha256"]].append(i)
    reps = []
    for idx in by_sha.values():
        reps.append(idx[0])
        for j in idx[1:]:
            union(j, idx[0])
            reason[j] = "точная копия"

    # 2. похожие кадры среди уникальных по хешу
    E = np.stack([items[i]["emb"] for i in reps])
    S = E @ E.T
    P = np.array([items[i]["phash"] for i in reps], dtype=np.uint64)  # (m, 4 поворота)
    similar = []
    checked = 0
    for a in range(len(reps)):
        pd = popcount(P[a, 0] ^ P[a + 1:]).min(axis=1)  # лучший из 4 поворотов
        sims = S[a, a + 1:]
        for off in np.nonzero((sims >= CAND_CLIP) | (pd <= CAND_PHASH))[0]:
            b = a + 1 + off
            ia, ib = reps[a], reps[b]
            if find(ia) == find(ib):
                continue
            if pd[off] <= SURE_PHASH and sims[off] >= 0.85:
                ok, why = True, f"phash {int(pd[off])}"
            else:
                checked += 1
                ok, why = same_frame(items[ia], items[ib])
            if ok:
                union(ia, ib)
                reason[ib] = reason.get(ib) or f"тот же кадр ({why})"
                reason[ia] = reason.get(ia) or f"тот же кадр ({why})"
            elif sims[off] >= SIMILAR_REPORT:
                similar.append((ia, ib, float(sims[off])))
    log(f"  проверено геометрией пар: {checked}")

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    similar = [(a, b, s) for a, b, s in similar if find(a) != find(b)]
    return [g for g in groups.values() if len(g) > 1], reason, similar


def quality_key(it: dict):
    """Лучший в группе: больше пикселей, затем больше байт (меньше сжатие), затем резче."""
    return (it["width"] * it["height"], it["size"], it.get("sharp", 0), -len(it["rel"]), it["rel"])


# ---------------------------------------------------------------- категории


class Classifier:
    def __init__(self, clip: Clip):
        self.cat_keys = list(CATEGORIES)
        self.other_keys = list(OTHER)
        self.labels = self.cat_keys + self.other_keys
        self.M = clip.class_matrix([CATEGORIES[k][3] for k in self.cat_keys] +
                                   [OTHER[k] for k in self.other_keys])
        self.colors = {cat: clip.class_matrix([color_prompts(w, extra, noun, parts) for _, w, extra in COLORS])
                       for cat, (noun, parts) in COLOR_NOUN.items()}
        self.types = {cat: (list(t), clip.class_matrix(list(t.values()))) for cat, t in TYPES.items()}
        self.finish = (list(FINISH), clip.class_matrix(list(FINISH.values())))

    @staticmethod
    def softmax(emb: np.ndarray, M: np.ndarray) -> np.ndarray:
        z = 100.0 * (M @ emb)
        z = np.exp(z - z.max())
        return z / z.sum()

    def classify(self, emb: np.ndarray) -> dict:
        p = self.softmax(emb, self.M)
        order = np.argsort(-p)
        top = [(self.labels[i], float(p[i])) for i in order[:3]]
        best, conf = top[0]
        res = {"top": top, "conf": conf}
        if best in self.other_keys:
            res.update(cat=None, why=f"{best} ({conf:.0%})")
        elif conf < MIN_CONF or conf - top[1][1] < MIN_MARGIN:
            names = " или ".join(FOLDERS.get(k, k) for k, _ in top[:2])
            res.update(cat=None, why=f"не уверен: {names} ({conf:.0%}/{top[1][1]:.0%})")
        else:
            res.update(cat=best, why="")
        guess = next((k for k, _ in top if k in CATEGORIES), None)
        res["guess"] = guess
        res["desc"] = self.describe(emb, res["cat"] or guess) if guess else ""
        return res

    def describe(self, emb: np.ndarray, cat: str) -> str:
        parts = []
        gender = 1 if CATEGORIES[cat][2] == "m" else 0
        if cat in self.colors:
            by_slug = defaultdict(float)
            for (slugs, _, _), v in zip(COLORS, self.softmax(emb, self.colors[cat])):
                by_slug[slugs[gender]] += float(v)
            color = max(by_slug, key=by_slug.get)
            if by_slug[color] >= 0.5:
                parts.append(color)
        slugs, M = self.types[cat]
        pt = self.softmax(emb, M)
        if pt.max() >= 0.5:
            parts.append(slugs[int(pt.argmax())])
        if cat != "project":
            fs, FM = self.finish
            pf = self.softmax(emb, FM)
            f = fs[int(pf.argmax())]
            if f and pf.max() >= 0.60 and not (f == "shpon" and "pod-derevo" in parts):
                parts.append(f)
        return "-".join(parts)


# ---------------------------------------------------------------- analyze

CSV_COLS = ["файл", "статус", "категория", "описание", "новое_имя", "уверенность", "причина",
            "группа_дублей", "дубль_чего", "проблемы", "разрешение", "размер_мб", "варианты"]
STATUS_KEEP, STATUS_UNSORTED, STATUS_DUP = "в портфолио", "разобрать", "дубль"
STATUS_BROKEN, STATUS_NOTPHOTO, STATUS_EXCLUDE = "битый", "не фото", "исключить"


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "rb") as fh:
                cache = pickle.load(fh)
            if cache.get("model") == MODEL and cache.get("version") == CACHE_VERSION:
                return cache
        except Exception:
            pass
    return {"model": MODEL, "version": CACHE_VERSION, "items": {}}


def analyze(src: Path, svc: Path, limit: int | None = None) -> None:
    t0 = time.time()
    svc.mkdir(exist_ok=True)
    log(f"\n=== 1. Инвентаризация: {src}")
    files = scan(src, read_journal(svc))
    inv = inventory_summary(files)
    print_inventory(inv)

    images = [f for f in files if f["kind"] == "image"][:limit]
    cache_path = svc / "кэш.pkl"
    cache = load_cache(cache_path)
    clip = Clip(svc / "models")

    log(f"\n=== Анализ {len(images)} фото (хеши, CLIP, ORB)")
    items, todo = [], []
    for f in images:
        c = cache["items"].get(f["rel"])
        if c and c["size"] == f["size"] and c["mtime"] == f["mtime"]:
            items.append(c)
        else:
            todo.append(f)
    if len(items):
        log(f"  из кэша: {len(items)}")
    workers = min(8, os.cpu_count() or 2)
    done, t1 = 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start in range(0, len(todo), 32):
            chunk = list(ex.map(lambda r: safe_extract(src, r, clip.preprocess), todo[start:start + 32]))
            ok = [it for it in chunk if "broken" not in it]
            if ok:
                emb = clip.encode_images([t for it in ok for t in it.pop("_tensors")])
                emb = emb.reshape(len(ok), 2, -1).mean(1)
                emb /= np.linalg.norm(emb, axis=1, keepdims=True)
                for it, e in zip(ok, emb):
                    it["emb"] = e
            for it in chunk:
                cache["items"][it["rel"]] = it
            items.extend(chunk)
            done += len(chunk)
            rate = (time.time() - t1) / done
            log(f"  {done}/{len(todo)}  ~{rate * (len(todo) - done) / 60:.0f} мин осталось")
            if start % 320 == 0:
                with open(cache_path, "wb") as fh:
                    pickle.dump(cache, fh)
    with open(cache_path, "wb") as fh:
        pickle.dump(cache, fh)

    items.sort(key=lambda it: it["rel"])
    good = [it for it in items if "broken" not in it]
    broken = [it for it in items if "broken" in it]
    log(f"  читаемых: {len(good)}, битых: {len(broken)}")

    log("\n=== 2. Дубли")
    groups, dup_reason, similar = find_duplicates(good)
    dup_of, group_id = {}, {}
    for gi, g in enumerate(sorted(groups, key=lambda g: min(good[i]["rel"] for i in g)), 1):
        best = max(g, key=lambda i: quality_key(good[i]))
        for i in g:
            group_id[i] = gi
            if i != best:
                dup_of[i] = best
                if good[i]["sha256"] == good[best]["sha256"]:
                    dup_reason[i] = "точная копия"
                else:
                    dup_reason[i] = next((dup_reason[k] for k in (i, best) if dup_reason.get(k, "").startswith("тот")),
                                         "тот же кадр")
        dup_reason.setdefault(best, "оставлен")
        if dup_reason[best] == "точная копия":
            dup_reason[best] = "оставлен"
    exact = sum(1 for i in dup_of if dup_reason.get(i) == "точная копия")
    log(f"  групп: {len(groups)}, в _дубли уйдёт: {len(dup_of)} (точных копий: {exact}, "
        f"похожих кадров: {len(dup_of) - exact})")

    log("\n=== 3. Категории")
    clf = Classifier(clip)
    rows = []
    for i, it in enumerate(good):
        res = clf.classify(it["emb"])
        problems = []
        if it["lum"] < DARK_LUM:
            problems.append(f"тёмное (яркость {it['lum']:.0f})")
        if it["sharp"] < BLUR_VAR:
            problems.append(f"возможно размыто (резкость {it['sharp']:.0f})")
        if max(it["width"], it["height"]) < SMALL_SIDE:
            problems.append("низкое разрешение")
        if i in dup_of:
            status = STATUS_DUP
        elif res["cat"]:
            status = STATUS_KEEP
        else:
            status = STATUS_UNSORTED
        cat = res["cat"] or (res["guess"] if status == STATUS_DUP else None)
        rows.append({
            "файл": it["rel"], "статус": status,
            "категория": FOLDERS[cat] if cat else UNSORTED,
            "описание": res["desc"], "новое_имя": "",
            "уверенность": f"{res['conf']:.2f}",
            "причина": res["why"] if status == STATUS_UNSORTED else "",
            "группа_дублей": group_id.get(i, ""),
            "дубль_чего": good[dup_of[i]]["rel"] if i in dup_of else "",
            "проблемы": "; ".join(problems),
            "разрешение": f"{it['width']}x{it['height']}",
            "размер_мб": f"{it['size'] / 2**20:.2f}",
            "варианты": ", ".join(f"{FOLDERS.get(k, k)} {p:.0%}" for k, p in res["top"]),
            "_dupwhy": dup_reason.get(i, "") if i in group_id else "",
            "_descs": {FOLDERS[k]: clf.describe(it["emb"], k) for k in CATEGORIES},
            "_taken": it.get("taken", ""), "_gps": it.get("gps", False),
        })
    for it in broken:
        rows.append({"файл": it["rel"], "статус": STATUS_BROKEN, "категория": "", "описание": "",
                     "новое_имя": "", "уверенность": "", "причина": it["broken"], "группа_дублей": "",
                     "дубль_чего": "", "проблемы": "файл не читается",
                     "разрешение": "", "размер_мб": f"{it['size'] / 2**20:.2f}", "варианты": ""})
    for f in files:
        if f["kind"] != "image":
            rows.append({"файл": f["rel"], "статус": STATUS_NOTPHOTO, "категория": "", "описание": "",
                         "новое_имя": "", "уверенность": "", "причина": KIND_RU[f["kind"]], "группа_дублей": "",
                         "дубль_чего": "", "проблемы": "", "разрешение": "",
                         "размер_мб": f"{f['size'] / 2**20:.2f}", "варианты": ""})
    assign_names(rows)
    csv_path = svc / "разметка.csv"
    if csv_path.exists():  # могли править руками — не теряем
        backup = svc / f"разметка-{dt.datetime.fromtimestamp(csv_path.stat().st_mtime):%Y%m%d-%H%M%S}.csv"
        if not backup.exists():
            shutil.copy2(csv_path, backup)
            log(f"  прежняя разметка сохранена: {backup.name}")
    write_csv(csv_path, rows)

    meta = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "src": str(src), "inventory": inv, "model": list(MODEL),
        "exif": {r["файл"]: [r.get("_taken", ""), r.get("_gps", False)] for r in rows if "_taken" in r},
        "dupwhy": {r["файл"]: r["_dupwhy"] for r in rows if r.get("_dupwhy")},
        # исходная категория и описания под каждую категорию — если категорию поменяют руками
        "auto": {r["файл"]: {"cat": r["категория"], "desc": r["описание"], "descs": r["_descs"]}
                 for r in rows if "_descs" in r},
        "similar": [[good[a]["rel"], good[b]["rel"], round(s, 3)] for a, b, s in similar],
        "dims": {it["rel"]: [it["width"], it["height"]] for it in good},
    }
    (svc / "анализ.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = Counter(r["категория"] for r in rows if r["статус"] == STATUS_KEEP)
    log("  " + ", ".join(f"{FOLDERS[k]}: {counts.get(FOLDERS[k], 0)}" for k in CATEGORIES)
        + f", _разобрать: {sum(r['статус'] == STATUS_UNSORTED for r in rows)}")
    log(f"\nРазметка: {svc / 'разметка.csv'}  ({(time.time() - t0) / 60:.1f} мин)")


def assign_names(rows: list[dict], meta_exif: dict | None = None) -> None:
    """категория-описание-NN.jpg; нумерация внутри одинакового начала имени, по дате съёмки."""
    stems = defaultdict(list)
    for r in rows:
        r["новое_имя"] = ""
        if r["статус"] != STATUS_KEEP:
            continue
        key = FOLDER_TO_KEY[r["категория"].lower()]
        r["описание"] = slugify(r["описание"])
        stem = "-".join(filter(None, [CATEGORIES[key][1], r["описание"]]))
        stems[(r["категория"], stem)].append(r)
    for (_, stem), group in stems.items():
        exif = meta_exif or {}
        group.sort(key=lambda r: (r.get("_taken") or (exif.get(r["файл"]) or [""])[0] or "9999", r["файл"]))
        width = max(2, len(str(len(group))))
        for n, r in enumerate(group, 1):
            r["новое_имя"] = f"{stem}-{n:0{width}d}.jpg"


def write_csv(path: Path, rows: list[dict]) -> None:
    order = {STATUS_KEEP: 0, STATUS_UNSORTED: 1, STATUS_DUP: 2, STATUS_BROKEN: 3, STATUS_NOTPHOTO: 4,
             STATUS_EXCLUDE: 5}
    rows = sorted(rows, key=lambda r: (order.get(r["статус"], 9), r["категория"], r["новое_имя"], r["файл"]))
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:  # BOM + «;» — открывается в Excel
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, delimiter=";", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=";"))


# ---------------------------------------------------------------- apply


def save_web_copy(src_path: Path, dst_path: Path) -> tuple[int, int]:
    im = open_image(src_path, draft=MAX_SIDE)
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)  # только уменьшает
    tmp = dst_path.with_name(dst_path.name + ".tmp")
    im.save(tmp, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)  # без EXIF/GPS
    os.replace(tmp, dst_path)
    return im.size


def apply(src: Path, dst: Path, svc: Path) -> None:
    t0 = time.time()
    csv_path = svc / "разметка.csv"
    if not csv_path.exists():
        sys.exit(f"Нет {csv_path} — сначала запустите analyze")
    rows = read_csv(csv_path)
    meta = json.loads((svc / "анализ.json").read_text(encoding="utf-8"))

    # ручные правки: статус следует за колонкой «категория»
    for r in rows:
        if r["статус"] not in (STATUS_KEEP, STATUS_UNSORTED):
            continue
        cat = r["категория"].strip().lower()
        if cat in FOLDER_TO_KEY:
            r["статус"], r["причина"] = STATUS_KEEP, ""
            r["категория"] = FOLDERS[FOLDER_TO_KEY[cat]]
            auto = meta.get("auto", {}).get(r["файл"])
            if auto and r["категория"] != auto["cat"] and slugify(r["описание"]) == auto["desc"]:
                r["описание"] = auto["descs"].get(r["категория"], "")  # описание не правили — берём под новую категорию
        else:
            if cat not in ("", UNSORTED):
                log(f"  ! {r['файл']}: неизвестная категория «{r['категория']}» — в _разобрать")
                r["причина"] = f"неизвестная категория «{r['категория']}»"
            r["статус"], r["категория"] = STATUS_UNSORTED, UNSORTED
    assign_names(rows, meta.get("exif"))
    moved_before = read_journal(svc)

    # 1. Дубли -> _дубли/группа-NNN/ (перенос, с журналом для undo)
    log("\n=== Перенос дублей в _дубли")
    journal = svc / "перемещения.jsonl"
    moved = 0
    with open(journal, "a", encoding="utf-8") as jf:
        for r in rows:
            if r["статус"] != STATUS_DUP:
                continue
            s = src / r["файл"]
            if not s.exists():
                continue  # уже перенесён прошлым запуском
            group = int(r["группа_дублей"] or 0)
            d = unique_path(src / DUPS / f"группа-{group:03d}" / Path(r["файл"]).name)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(s), str(d))
            jf.write(json.dumps({"from": r["файл"], "to": d.relative_to(src).as_posix(),
                                 "at": dt.datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False) + "\n")
            moved += 1
    log(f"  перенесено сейчас: {moved}")

    # 2. Портфолио: прошлые копии скрипта -> _служебное/старые-версии (не удаляем)
    manifest_path = svc / "портфолио-манифест.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    plan = {(FOLDERS[FOLDER_TO_KEY[r["категория"].lower()]], r["новое_имя"]): r
            for r in rows if r["статус"] == STATUS_KEEP}
    stale = [p for p in old if tuple(p) not in plan and (dst / p[0] / p[1]).exists()]
    if stale:
        arch = svc / "старые-версии" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        for folder, name in stale:
            (arch / folder).mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst / folder / name), str(arch / folder / name))
        log(f"  прошлые копии, которых нет в новой разметке, убраны в {arch}")
    for folder in FOLDERS.values():
        (dst / folder).mkdir(parents=True, exist_ok=True)
    foreign = [p for p in dst.rglob("*") if p.is_file() and p.name != "ОТЧЁТ.txt"
               and (p.parent.name, p.name) not in plan and [p.parent.name, p.name] not in old]

    log(f"\n=== Копии для Портфолио ({len(plan)} шт., ≤{MAX_SIDE} px, JPEG {JPEG_QUALITY})")
    errors = []

    def work(item):
        (folder, name), r = item
        try:
            r["_out"] = save_web_copy(locate(src, r["файл"], moved_before), dst / folder / name)
        except Exception as e:
            errors.append((r["файл"], f"{type(e).__name__}: {e}"))

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        for k, _ in enumerate(ex.map(work, plan.items()), 1):
            if k % 100 == 0:
                log(f"  {k}/{len(plan)}")
    manifest_path.write_text(json.dumps([list(k) for k in plan], ensure_ascii=False), encoding="utf-8")

    # 3. _разобрать — копии оригиналов + список
    unsorted = [r for r in rows if r["статус"] == STATUS_UNSORTED]
    udir = src / UNSORTED
    udir.mkdir(exist_ok=True)
    umanifest_path = svc / "разобрать-манифест.json"
    umanifest = json.loads(umanifest_path.read_text(encoding="utf-8")) if umanifest_path.exists() else {}
    wanted = {r["файл"] for r in unsorted}
    for rel, name in list(umanifest.items()):
        if rel not in wanted:  # уже разложили руками
            if (udir / name).exists():
                arch = svc / "старые-версии" / dt.datetime.now().strftime("%Y%m%d-%H%M%S") / UNSORTED
                arch.mkdir(parents=True, exist_ok=True)
                shutil.move(str(udir / name), str(unique_path(arch / name)))
            del umanifest[rel]
    for r in unsorted:
        if r["файл"] in umanifest and (udir / umanifest[r["файл"]]).exists():
            continue
        d = unique_path(udir / Path(r["файл"]).name)
        shutil.copy2(locate(src, r["файл"], moved_before), d)
        umanifest[r["файл"]] = d.name
    umanifest_path.write_text(json.dumps(umanifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (udir / "список.txt").write_text(
        "Фото, которые не удалось уверенно отнести к категории (копии оригиналов).\n"
        "Чтобы разложить: в _служебное/разметка.csv поставить «категорию» и запустить apply.\n\n"
        + "\n".join(f"{r['файл']}\t{r['причина']}\t[{r['варианты']}]" for r in unsorted) + "\n",
        encoding="utf-8")

    write_csv(csv_path, rows)
    report = build_report(rows, meta, dst, errors, foreign)
    (dst / "ОТЧЁТ.txt").write_text(report, encoding="utf-8-sig")
    (svc / "ОТЧЁТ.txt").write_text(report, encoding="utf-8-sig")
    build_review_html(rows, meta, src, dst, svc / "проверка.html")
    log(f"\nГотово за {(time.time() - t0) / 60:.1f} мин.\n  Портфолио: {dst}\n  Отчёт: {dst / 'ОТЧЁТ.txt'}"
        f"\n  Проверка глазами: {svc / 'проверка.html'}")
    log("\n" + report.split("\n\n")[2] if report.count("\n\n") > 2 else "")


# ---------------------------------------------------------------- отчёт


def build_report(rows, meta, dst, errors, foreign) -> str:
    inv = meta["inventory"]
    by = defaultdict(list)
    for r in rows:
        by[r["статус"]].append(r)
    dupwhy = meta.get("dupwhy", {})
    exact = sum(1 for r in by[STATUS_DUP] if dupwhy.get(r["файл"]) == "точная копия")
    groups = {r["группа_дублей"] for r in rows if r["группа_дублей"]}
    keep = by[STATUS_KEEP]
    counts = Counter(r["категория"] for r in keep)
    L = []
    add = L.append
    add("ОТЧЁТ ПО РАЗБОРУ ПОРТФОЛИО")
    add(f"Дата: {dt.datetime.now():%d.%m.%Y %H:%M}. Источник: {meta['src']}")
    add("")
    add("1. ИСХОДНАЯ ПАПКА")
    k = inv["kinds"]
    add(f"Всего файлов: {inv['total']} ({human(inv['bytes'])}). Фото: {k.get('image', 0)}, "
        f"видео: {k.get('video', 0)}, прочее: {k.get('other', 0)}, системные: {k.get('system', 0)}.")
    add("Форматы: " + ", ".join(f"{e} — {n} ({human(s)})" for e, (n, s) in inv["by_ext"].items()))
    add("")
    add("2. ИТОГ")
    add(f"В Портфолио: {len(keep)} фото.")
    for key in CATEGORIES:
        n = counts.get(FOLDERS[key], 0)
        add(f"  {FOLDERS[key]:16s} {n:5d}" + ("   (пусто — папку можно удалить)" if n == 0 else ""))
    add(f"Дублей убрано в _дубли: {len(by[STATUS_DUP])} (точных копий: {exact}, "
        f"похожих кадров: {len(by[STATUS_DUP]) - exact}; групп: {len(groups)}).")
    add(f"Не удалось разложить (_разобрать): {len(by[STATUS_UNSORTED])}. "
        f"Исключено вручную: {len(by[STATUS_EXCLUDE])}. Битых: {len(by[STATUS_BROKEN])}. "
        f"Не фото: {len(by[STATUS_NOTPHOTO])}.")
    add("")
    add(f"3. НЕ УДАЛОСЬ РАЗЛОЖИТЬ ({len(by[STATUS_UNSORTED])}) — копии в «{UNSORTED}» рядом с оригиналами")
    for r in by[STATUS_UNSORTED]:
        add(f"  {r['файл']} — {r['причина']}")
    add("")
    add("4. ПРОБЛЕМЫ")
    if by[STATUS_BROKEN]:
        add(f"Битые файлы ({len(by[STATUS_BROKEN])}) — не открываются, в Портфолио не попали:")
        for r in by[STATUS_BROKEN]:
            add(f"  {r['файл']} — {r['причина']}")
    for title, test in (("Очень тёмные", "тёмное"), ("Возможно размытые", "размыто"),
                        ("Низкое разрешение (длинная сторона < %d px)" % SMALL_SIDE, "низкое")):
        hit = [r for r in keep if test in r["проблемы"]]
        if hit:
            add(f"{title} ({len(hit)}) — в Портфолио есть, стоит глянуть:")
            for r in hit:
                add(f"  {r['категория']}/{r['новое_имя']}  ←  {r['файл']}  [{r['разрешение']}]")
    if errors:
        add(f"Ошибки при сохранении копий ({len(errors)}):")
        for f, e in errors:
            add(f"  {f} — {e}")
    if by[STATUS_NOTPHOTO]:
        add(f"Не фото ({len(by[STATUS_NOTPHOTO])}) — не обрабатывались, лежат на месте:")
        for r in by[STATUS_NOTPHOTO]:
            add(f"  {r['файл']} ({r['причина']}, {r['размер_мб']} МБ)")
    gps = sum(1 for v in meta.get("exif", {}).values() if v[1])
    if gps:
        add(f"В {gps} оригиналах записаны GPS-координаты (адреса клиентов). "
            f"В копиях для Портфолио все метаданные удалены.")
    if meta.get("similar"):
        add(f"Очень похожих, но разных кадров: {len(meta['similar'])} пар — оставлены оба, "
            f"см. _служебное/проверка.html.")
    if foreign:
        add(f"В папке Портфолио есть файлы, созданные не скриптом ({len(foreign)}) — не тронуты:")
        for p in foreign[:50]:
            add(f"  {p.relative_to(dst)}")
    add("")
    add("5. ПРИНЯТЫЕ РЕШЕНИЯ")
    for line in DECISIONS:
        add(f"- {line}")
    return "\n".join(L) + "\n"


DECISIONS = [
    "Категория определяется по изображению (нейросеть CLIP, сравнение картинки с описаниями категорий), "
    "имя файла не учитывается.",
    "Дубли: точные — по SHA-256; похожие — кандидаты по perceptual hash (с учётом поворотов) и смысловой "
    "близости CLIP, затем проверка «тот же кадр»: ORB-точки совпадают через поворот/масштаб/сдвиг, один кадр "
    "почти целиком лежит в другом и после совмещения совпадают мелкие детали (кроп, пережатие, поворот, "
    "уменьшение, правка яркости). Разные ракурсы одной мебели дублями не считаются.",
    "Из группы дублей остаётся фото с наибольшим разрешением, при равенстве — самый большой файл "
    "(меньше сжатие). Остальные перенесены в «_дубли/группа-NNN» (не удалены). Журнал переноса — "
    "_служебное/перемещения.jsonl, вернуть всё: команда undo.",
    "Оригиналы не переименовывались и не перемещались (кроме дублей). Понятные имена получили копии в Портфолио; "
    "соответствие «оригинал → новое имя» — в _служебное/разметка.csv.",
    "«_разобрать» лежит рядом с оригиналами, а не в Портфолио: Портфолио уходит клиентам по ссылке.",
    "Тумба под раковиной относится к «Ванным», обувница/вешалка — к «Прихожим», тумбы под ТВ/комоды — к «Тумбам».",
    "«Проекты квартир» — планировки, визуализации, коллажи и кадры, где видно несколько зон (кухня-гостиная).",
    "Неуверенные случаи (уверенность < %d%% или почти равные две категории), скриншоты, чертежи, люди, "
    "мастерская, другая мебель (кровати, столы) — в «_разобрать»." % round(MIN_CONF * 100),
    "Описание в имени (цвет, тип, глянец/шпон) — то, что модель видит с уверенностью; "
    "если не уверена — слово не пишется. Нумерация — по дате съёмки.",
    "Копии: длинная сторона ≤ %d px (меньшие не увеличивались), JPEG %d, ориентация по EXIF применена, "
    "цвет приведён к sRGB, метаданные удалены." % (MAX_SIDE, JPEG_QUALITY),
    "Тёмные/размытые/маленькие фото не исключались — только перечислены выше: решение за человеком.",
    "Этот отчёт лежит и в Портфолио (по заданию), и в _служебное. Перед отправкой ссылки клиентам "
    "его стоит убрать из Портфолио.",
]


def build_review_html(rows, meta, src, dst, out: Path) -> None:
    def img(path: Path, caption: str, cls: str = "") -> str:
        return (f'<figure class="{cls}"><img loading="lazy" src="{path.resolve().as_uri()}">'
                f"<figcaption>{html.escape(caption)}</figcaption></figure>")

    parts = []
    keep = [r for r in rows if r["статус"] == STATUS_KEEP]
    for key in CATEGORIES:
        folder = FOLDERS[key]
        rs = sorted((r for r in keep if r["категория"] == folder), key=lambda r: r["новое_имя"])
        parts.append(f"<h2>{folder} — {len(rs)}</h2><div class=grid>")
        parts += [img(dst / folder / r["новое_имя"], f"{r['новое_имя']} · {r['уверенность']}\n{r['файл']}",
                      "low" if float(r["уверенность"] or 0) < 0.6 else "") for r in rs]
        parts.append("</div>")
    rs = [r for r in rows if r["статус"] == STATUS_UNSORTED]
    parts.append(f"<h2>_разобрать — {len(rs)}</h2><div class=grid>")
    parts += [img(src / r["файл"], f"{r['файл']}\n{r['причина']}\n{r['варианты']}") for r in rs]
    parts.append("</div>")

    journal = read_journal(out.parent)
    by_group = defaultdict(list)
    for r in rows:
        if r["группа_дублей"]:
            by_group[int(r["группа_дублей"])].append(r)
    parts.append(f"<h2>Дубли — {len(by_group)} групп (рамка — оставленный)</h2>")
    dupwhy = meta.get("dupwhy", {})
    for g, rs in sorted(by_group.items()):
        parts.append(f"<h3>группа-{g:03d}</h3><div class=grid>")
        for r in sorted(rs, key=lambda r: r["статус"] == STATUS_DUP):
            is_dup = r["статус"] == STATUS_DUP
            here = src / journal.get(r["файл"], r["файл"])
            if r["статус"] == STATUS_KEEP:
                here = dst / r["категория"] / r["новое_имя"]
            parts.append(img(here, f"{r['файл']} {r['разрешение']}\n{dupwhy.get(r['файл'], '')}",
                             "" if is_dup else "kept"))
        parts.append("</div>")
    if meta.get("similar"):
        parts.append(f"<h2>Похожие, но оставлены оба — {len(meta['similar'])} пар</h2>")
        for a, b, s in meta["similar"]:
            pa, pb = src / journal.get(a, a), src / journal.get(b, b)
            parts.append("<div class=grid>" + img(pa, a) + img(pb, f"{b}\nCLIP {s}") + "</div>")

    out.write_text(
        "<!doctype html><meta charset=utf-8><title>Проверка разбора</title><style>"
        "body{font:14px system-ui;margin:16px;background:#f4f4f4}"
        ".grid{display:flex;flex-wrap:wrap;gap:8px}figure{margin:0;width:220px;background:#fff;padding:4px}"
        "img{width:220px;height:165px;object-fit:cover;display:block}"
        "figcaption{font-size:11px;white-space:pre-wrap;word-break:break-all;color:#333}"
        ".kept{outline:3px solid #2a9d4b}.low{outline:2px dashed #e08a00}"
        "</style><h1>Проверка разбора</h1><p>Оранжевая пунктирная рамка — уверенность ниже 60%. "
        "Исправления: _служебное/разметка.csv → колонка «категория» → команда apply.</p>"
        + "\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------- undo


def undo(src: Path, svc: Path) -> None:
    jp = svc / "перемещения.jsonl"
    if not jp.exists():
        sys.exit("Журнал перемещений пуст — возвращать нечего")
    back = 0
    for line in reversed(jp.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        j = json.loads(line)
        s, d = src / j["to"], src / j["from"]
        if s.exists() and not d.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(s), str(d))
            back += 1
    jp.rename(jp.with_name(f"перемещения-отменено-{dt.datetime.now():%Y%m%d-%H%M%S}.jsonl"))
    log(f"Возвращено на место: {back}. Портфолио и _разобрать не тронуты (это копии).")


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "analyze", "apply", "undo"])
    ap.add_argument("--src", type=Path, help="папка с фото (по умолчанию — та, где лежит _служебное "
                                             "со скриптом, иначе %s)" % DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, help="папка Портфолио (по умолчанию — «Портфолио» рядом с src)")
    ap.add_argument("--limit", type=int, help="обработать только N фото (для пробы)")
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    src = (a.src or (here.parent if here.name == SERVICE else DEFAULT_SRC)).resolve()
    dst = (a.dst or src.parent / "Портфолио").resolve()
    if not src.is_dir():
        sys.exit(f"Нет папки {src}")
    if dst == src or src in dst.parents:
        sys.exit("Портфолио не должно лежать внутри папки с оригиналами")
    svc = src / SERVICE
    if a.command in ("run", "analyze"):
        analyze(src, svc, a.limit)
    if a.command in ("run", "apply"):
        apply(src, dst, svc)
    if a.command == "undo":
        undo(src, svc)


if __name__ == "__main__":
    main()
