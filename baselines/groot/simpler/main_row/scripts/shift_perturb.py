"""Shift-aware track (SHIFT_2026-11-03.md §1): photometric and geometric copies of a 256x256 observation. One parameter draw per copy
is applied identically to both camera views. rng = default_rng(SeedSequence([20261103, stream, i, j])); stream 0 = calibration,
stream 1 = held-out evaluation. Copies j = 1, 2 photometric, j = 3 geometric (j = 4 on-policy is not an image transform)."""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SEED = 20261103
KINDS = {1: 'photometric', 2: 'photometric', 3: 'geometric'}


def rng_for(stream, i, j):
    return np.random.default_rng(np.random.SeedSequence([SEED, stream, i, j]))


def photometric_params(rng):
    return dict(brightness=float(rng.uniform(0.8, 1.2)), contrast=float(rng.uniform(0.8, 1.2)), hue_deg=float(rng.uniform(-5.0, 5.0)), blur_sigma=float(rng.uniform(0.5, 1.0)))


def geometric_params(rng, size=256):
    area = float(rng.uniform(0.90, 0.95))
    side = int(round(np.sqrt(area) * size))
    top = int(rng.integers(0, size - side + 1)); left = int(rng.integers(0, size - side + 1))
    return dict(area=area, side=side, top=top, left=left)


def apply_photometric(img, p):
    im = Image.fromarray(np.asarray(img, dtype=np.uint8))
    im = ImageEnhance.Brightness(im).enhance(p['brightness'])
    im = ImageEnhance.Contrast(im).enhance(p['contrast'])
    hsv = np.asarray(im.convert('HSV')).copy()
    shift = int(round(p['hue_deg'] / 360.0 * 256.0))
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 256).astype(np.uint8)
    im = Image.fromarray(hsv, 'HSV').convert('RGB')
    im = im.filter(ImageFilter.GaussianBlur(radius=p['blur_sigma']))
    return np.ascontiguousarray(np.asarray(im, dtype=np.uint8))


def apply_geometric(img, p):
    im = Image.fromarray(np.asarray(img, dtype=np.uint8))
    h, w = im.size[1], im.size[0]
    im = im.crop((p['left'], p['top'], p['left'] + p['side'], p['top'] + p['side'])).resize((w, h), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(im, dtype=np.uint8))


def perturbed_copy(obs, stream, i, j):
    """obs: bridge observation dict ('observation/image', 'observation/wrist_image', ...). Returns (new obs, record)."""
    kind = KINDS[j]
    rng = rng_for(stream, i, j)
    if kind == 'photometric':
        p = photometric_params(rng); f = apply_photometric
    else:
        p = geometric_params(rng, np.asarray(obs['observation/image']).shape[0]); f = apply_geometric
    out = dict(obs)
    out['observation/image'] = f(obs['observation/image'], p)
    out['observation/wrist_image'] = f(obs['observation/wrist_image'], p)
    return out, dict(kind=kind, stream=stream, i=i, j=j, params=p)
