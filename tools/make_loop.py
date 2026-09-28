#!/usr/bin/env python3
"""
Зацикленное видео смены света: каждое состояние держится секунду,
переход — плавный кросс-фейд. Последний кадр перетекает в первый,
поэтому петля стыкуется без рывка.

Запуск:  python3 tools/make_loop.py images video
"""
import pathlib
import subprocess
import sys

import cv2
import numpy as np

ORDER = ['day', 'sunset', 'evening']   # порядок состояний
FPS = 30
BEAT = 1.0          # секунда на состояние
FADE = 0.35         # из них на переход
WIDTH = 1360        # ширина видео (должна быть чётной)


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'


def smoothstep(t):
    """Плавное ускорение-замедление — переход не начинается и не кончается рывком."""
    return t * t * (3 - 2 * t)


def build_frames(src_dir):
    imgs = []
    for name in ORDER:
        path = src_dir / f'{name}.webp'
        if not path.exists():
            raise SystemExit(f'нет файла {path}')
        im = cv2.imread(str(path)).astype(np.float32)
        imgs.append(im)
    h, w = imgs[0].shape[:2]
    for im in imgs[1:]:
        if im.shape[:2] != (h, w):
            raise SystemExit('кадры разного размера — сначала прогоните tools/align.py')

    beat = int(round(BEAT * FPS))
    fade = int(round(FADE * FPS))
    hold = beat - fade
    for i, cur in enumerate(imgs):
        nxt = imgs[(i + 1) % len(imgs)]     # последнее состояние перетекает в первое
        for _ in range(hold):
            yield cur
        for f in range(fade):
            # знаменатель fade+1: альфа не доходит до единицы, иначе последний
            # кадр перехода совпал бы с первым кадром следующего состояния
            # и петля дёргалась бы на каждом стыке
            a = smoothstep((f + 1) / (fade + 1))
            yield cur * (1 - a) + nxt * a


def main(src_dir='images', out_dir='video'):
    src_dir, out_dir = pathlib.Path(src_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_bin()

    raw = out_dir / '_frames.mkv'          # промежуточный файл без потерь
    frames = list(build_frames(src_dir))
    h, w = frames[0].shape[:2]
    scale = WIDTH / w
    size = (WIDTH, int(round(h * scale)) // 2 * 2)   # обе стороны чётные — требование H.264
    print(f'{len(frames)} кадров, {len(frames) / FPS:.1f} с, {size[0]}x{size[1]}')

    writer = subprocess.Popen(
        [ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
         '-s', f'{size[0]}x{size[1]}', '-r', str(FPS), '-i', 'pipe:0',
         '-c:v', 'ffv1', str(raw)],
        stdin=subprocess.PIPE)
    for fr in frames:
        small = cv2.resize(np.clip(fr, 0, 255).astype(np.uint8), size,
                           interpolation=cv2.INTER_AREA)
        writer.stdin.write(small.tobytes())
    writer.stdin.close()
    writer.wait()

    encodes = [
        ('light.mp4',  ['-c:v', 'libx264', '-preset', 'veryslow', '-crf', '20',
                        '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.0',
                        '-g', str(FPS), '-movflags', '+faststart', '-an']),
        ('light.webm', ['-c:v', 'libvpx-vp9', '-crf', '32', '-b:v', '0',
                        '-row-mt', '1', '-pix_fmt', 'yuv420p', '-an']),
    ]
    for name, args in encodes:
        subprocess.run([ff, '-y', '-loglevel', 'error', '-i', str(raw), *args,
                        str(out_dir / name)], check=True)

    # постер — первый кадр, показывается пока видео грузится
    subprocess.run([ff, '-y', '-loglevel', 'error', '-i', str(raw), '-frames:v', '1',
                    '-c:v', 'libwebp', '-quality', '86', str(out_dir / 'poster.webp')],
                   check=True)
    raw.unlink()

    for f in sorted(out_dir.iterdir()):
        print(f'  {f}  {f.stat().st_size / 1024:.0f} КБ')


if __name__ == '__main__':
    main(*sys.argv[1:3] if len(sys.argv) > 2 else ('images', 'video'))
