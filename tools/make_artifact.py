#!/usr/bin/env python3
"""Собирает из index.html фрагмент для публикации в Artifacts (без doctype/html/head/body)."""
import re, sys, pathlib

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'index.html').read_text(encoding='utf-8')


def between(open_tag, close_tag):
    m = re.search(re.escape(open_tag) + r'(.*?)' + re.escape(close_tag), src, re.S)
    if not m:
        raise SystemExit(f'не найден блок {open_tag}')
    return m.group(1).strip('\n')


out = between('<!--artifact:head-->', '<!--/artifact:head-->') + '\n\n' + \
      between('<!--artifact:body-->', '<!--/artifact:body-->') + '\n'
dest = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else 'artifact.html')
dest.write_text(out, encoding='utf-8')
print(f'{dest} — {len(out)} символов')
