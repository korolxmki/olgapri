#!/usr/bin/env python3
"""Собирает self-contained index.html: подставляет фото из assets/ как data-URI."""
import base64
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
TEMPLATE = ROOT / "src" / "template.html"
OUTPUT = ROOT / "index.html"

IMAGES = {
    "HERO_BG": "hero-wardrobe.jpg",     # фон hero, Ken Burns
    "HERO_PANEL": "wardrobe-open.jpg",  # правая панель hero
    "LOUVERED": "louvered-detail.jpg",  # фон featured-карточки услуг
    "KITCHEN": "kitchen-island.jpg",    # секция «Почему мы»
}


def data_uri(path: pathlib.Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    html = TEMPLATE.read_text(encoding="utf-8")
    for token, filename in IMAGES.items():
        src = ROOT / "assets" / filename
        if not src.exists():
            print(f"ОШИБКА: нет файла {src}", file=sys.stderr)
            return 1
        placeholder = "{{" + token + "}}"
        if placeholder not in html:
            print(f"ОШИБКА: в шаблоне нет {placeholder}", file=sys.stderr)
            return 1
        html = html.replace(placeholder, data_uri(src))
        print(f"  {token:11s} <- assets/{filename} ({src.stat().st_size // 1024} KB)")

    if "{{" in html:
        print("ОШИБКА: в шаблоне остались незаполненные токены", file=sys.stderr)
        return 1

    OUTPUT.write_text(html, encoding="utf-8")
    print(f"\nГотово: {OUTPUT.name} — {OUTPUT.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
