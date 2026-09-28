#!/usr/bin/env python3
"""
Выравнивание серии фото одного интерьера в один кадр.

Задача: три снимка сняты/сгенерированы с чуть разных точек, поэтому при
переключении "прыгает" вся композиция. Скрипт приводит их к общей геометрии,
обрезает по максимальному общему прямоугольнику и сохраняет кадры одинакового
размера — при переключении меняется только свет.

Как работает:
  1. SIFT + RANSAC — гомография каждого кадра к опорному (перебор параметров,
     выбор по NCC на картинках с подавленной разницей освещения).
  2. Опорный кадр выбирается автоматически — тот, что даёт лучшую сумму NCC.
  3. Все кадры проецируются в систему координат опорного.
  4. Ищется наибольший вписанный прямоугольник заданного соотношения сторон,
     целиком лежащий в пересечении валидных областей всех кадров.

Специально НЕ используется нежёсткая деформация (optical flow): по метрикам
она лучше, но "плавит" прямые линии фасадов — на глянцевой кухне это видно сразу.

Запуск:  python3 tools/align.py source images
"""
import sys, pathlib
import cv2
import numpy as np

ASPECT = 4 / 3          # соотношение сторон результата
LONG_SIDE = 1360        # ширина итогового кадра, px
WIDTHS = (1360, 680)    # размеры для srcset
QUALITY = 92            # качество WebP


def normalize(img, sigma=21.0):
    """Локальная нормализация яркости: убирает разницу освещения, оставляет структуру."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.GaussianBlur(g, (0, 0), sigma)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur((g - mu) ** 2, (0, 0), sigma), 1.0))
    return np.clip((g - mu) / sd, -4, 4).astype(np.float32)


def ncc(a, b, mask):
    x, y = a[mask], b[mask]
    if x.size < 1000:
        return -1.0
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


def score(H, src_norm, ref_norm):
    h, w = ref_norm.shape
    warped = cv2.warpPerspective(src_norm, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpPerspective(np.ones_like(src_norm), H, (w, h), flags=cv2.INTER_NEAREST) > 0.5
    return ncc(ref_norm, warped, mask)


def best_homography(kd_src, kd_ref, src_norm, ref_norm):
    """Перебор порогов матчинга и RANSAC, выбор варианта с лучшим NCC."""
    (k1, d1), (k2, d2) = kd_src, kd_ref
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    best = None
    for ratio in (0.70, 0.75, 0.80, 0.85):
        good = [p for p, q in matches if p.distance < ratio * q.distance]
        if len(good) < 10:
            continue
        src = np.float32([k1[p.queryIdx].pt for p in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[p.trainIdx].pt for p in good]).reshape(-1, 1, 2)
        for thresh in (2.5, 3.5, 5.0):
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh,
                                         maxIters=200_000, confidence=0.9999)
            if H is None:
                continue
            s = score(H.astype(np.float32), src_norm, ref_norm)
            if best is None or s > best[0]:
                best = (s, H.astype(np.float32), int(mask.sum()))
    return best


def largest_inner_rect(valid, aspect):
    """Наибольший прямоугольник заданного соотношения внутри валидной области."""
    h, w = valid.shape
    integral = np.pad((valid < 255).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def fits(rw, rh):
        if rw > w or rh > h:
            return None
        s = (integral[rh:, rw:] - integral[:-rh, rw:]
             - integral[rh:, :-rw] + integral[:-rh, :-rw])
        ys, xs = np.nonzero(s == 0)
        if len(ys) == 0:
            return None
        # из всех подходящих позиций берём самую центральную
        d = (xs + rw / 2 - w / 2) ** 2 + (ys + rh / 2 - h / 2) ** 2
        i = int(np.argmin(d))
        return int(xs[i]), int(ys[i])

    lo, hi, found = 10, w, None
    while lo <= hi:                       # бинарный поиск по ширине
        mid = (lo + hi) // 2
        pos = fits(mid, int(round(mid / aspect)))
        if pos:
            found = (mid, int(round(mid / aspect)), *pos)
            lo = mid + 1
        else:
            hi = mid - 1
    if found is None:
        raise RuntimeError('общая область слишком мала — кадры не пересекаются')
    return found


def main(src_dir, out_dir):
    src_dir, out_dir = pathlib.Path(src_dir), pathlib.Path(out_dir)
    paths = sorted(src_dir.glob('*.webp')) + sorted(src_dir.glob('*.jpg')) + \
            sorted(src_dir.glob('*.png'))
    if len(paths) < 2:
        raise SystemExit(f'нужно минимум 2 изображения в {src_dir}')
    names = [p.stem for p in paths]
    imgs = {p.stem: cv2.imread(str(p)) for p in paths}
    norms = {n: normalize(v) for n, v in imgs.items()}

    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02, edgeThreshold=16)
    kd = {n: sift.detectAndCompute(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY), None)
          for n, v in imgs.items()}

    # опорный кадр — тот, к которому остальные ложатся лучше всего
    table, totals = {}, {}
    for ref in names:
        table[ref] = {ref: np.eye(3, dtype=np.float32)}
        total = 0.0
        for n in names:
            if n == ref:
                continue
            s, H, inliers = best_homography(kd[n], kd[ref], norms[n], norms[ref])
            table[ref][n] = H
            total += s
            print(f'  опорный {ref} <- {n}: NCC {s:.4f} (инлаеров {inliers})')
        totals[ref] = total
        print(f'опорный {ref}: сумма {total:.4f}')
    ref = max(totals, key=totals.get)
    print(f'\nопорный кадр: {ref}')

    # рендер в системе координат опорного, масштаб — по самому крупному исходнику
    scale = max(i.shape[1] for i in imgs.values()) / imgs[ref].shape[1]
    K = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
    W = int(round(imgs[ref].shape[1] * scale))
    H_ = int(round(imgs[ref].shape[0] * scale))

    warped, valid = {}, np.full((H_, W), 255, np.uint8)
    for n in names:
        T = (K @ table[ref][n]).astype(np.float32)
        warped[n] = cv2.warpPerspective(imgs[n], T, (W, H_), flags=cv2.INTER_LANCZOS4)
        v = cv2.warpPerspective(np.full(imgs[n].shape[:2], 255, np.uint8), T, (W, H_),
                                flags=cv2.INTER_NEAREST)
        valid &= cv2.erode(v, np.ones((5, 5), np.uint8))   # срезаем кайму интерполяции

    cw, ch, cx, cy = largest_inner_rect(valid, ASPECT)
    print(f'общий кроп: {cw}x{ch} @ ({cx},{cy}) — {100 * cw * ch / (W * H_):.0f}% холста')

    out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        crop = warped[n][cy:cy + ch, cx:cx + cw]
        for width in WIDTHS:
            size = (width, int(round(width / ASPECT)))
            interp = cv2.INTER_AREA if size[0] < cw else cv2.INTER_LANCZOS4
            suffix = '' if width == LONG_SIDE else f'-{width}'
            out = out_dir / f'{n}{suffix}.webp'
            cv2.imwrite(str(out), cv2.resize(crop, size, interpolation=interp),
                        [cv2.IMWRITE_WEBP_QUALITY, QUALITY])
            print(f'  {out} {size[0]}x{size[1]}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'source',
         sys.argv[2] if len(sys.argv) > 2 else 'images')
