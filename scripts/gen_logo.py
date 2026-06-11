"""Generate Murdock's icon/logo PNGs from the canonical geometry.

The mark: a devil's head drawn as a crimson circle with two horns
(Matt Murdock / Daredevil nod) filled with three rising sonar arcs —
Daredevil perceives the world through sound, Murdock identifies people
through it. Palette matches the Web UI (#dc2626 on #0c0f14 noir).

Outputs (run from the repo root, needs Pillow):
    icon.png                          256x256  HA add-on icon
    logo.png                          600x240  HA add-on logo (wordmark)
    murdock/ui/static/favicon.png      64x64   Web UI favicon

Everything is drawn supersampled (4x) and downscaled with Lanczos for
clean edges. The base geometry lives in 256x256 design units.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

BG = (12, 15, 20, 255)        # --bg     #0c0f14
CRIMSON = (220, 38, 38, 255)  # --accent #dc2626
BRIGHT = (239, 68, 68, 255)   # --danger #ef4444
DEEP = (153, 27, 27, 255)     # --accent-soft #991b1b
MUTED = (113, 113, 122, 255)  # --muted  #71717a

# Base design (256x256 units): head circle, horns, sonar arcs.
HEAD_C = (128.0, 142.0)
HEAD_R = 64.0
HEAD_SW = 13.0
# Each horn: (base-on-circle, ctrl-out, tip, ctrl-in, base-on-circle).
# Short outward-curving crescents — devil horns, not cat or rabbit ears.
HORN_L = ((80.0, 100.0), (58.0, 84.0), (62.0, 44.0), (84.0, 62.0), (104.0, 82.0))
HORN_R = ((176.0, 100.0), (198.0, 84.0), (194.0, 44.0), (172.0, 62.0), (152.0, 82.0))
ARC_C = (128.0, 188.0)
ARCS = (  # (radius, color, alpha)
    (18.0, BRIGHT, 255),
    (36.0, CRIMSON, 217),
    (54.0, DEEP, 255),
)
ARC_SW = 8.0
ARC_START, ARC_END = 200.0, 340.0  # PIL degrees, clockwise from 3 o'clock


def _quad(p0, c, p1, n=28):
    """Sample a quadratic Bezier."""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t ** 2 * p1[1]
        pts.append((x, y))
    return pts


def draw_mark(img: Image.Image, ox: float, oy: float, k: float,
              with_glow: bool = True) -> None:
    """Draw the head mark onto ``img`` at offset (ox, oy), scale ``k``."""
    def T(p):
        return (ox + k * p[0], oy + k * p[1])

    if with_glow:
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for r, alpha in ((100, 22), (70, 28)):
            cx, cy = T(HEAD_C)
            gd.ellipse(
                [cx - k * r, cy - k * r, cx + k * r, cy + k * r],
                fill=CRIMSON[:3] + (alpha,),
            )
        glow = glow.filter(ImageFilter.GaussianBlur(k * 9))
        img.alpha_composite(glow)

    d = ImageDraw.Draw(img)
    # Head circle (outline).
    cx, cy = T(HEAD_C)
    r = k * HEAD_R
    d.ellipse([cx - r, cy - r, cx + r, cy + r],
              outline=CRIMSON, width=max(1, round(k * HEAD_SW)))
    # Horns.
    for horn in (HORN_L, HORN_R):
        p0, c1, tip, c2, p1 = horn
        pts = _quad(p0, c1, tip) + _quad(tip, c2, p1)
        d.polygon([T(p) for p in pts], fill=CRIMSON)
    # Sonar arcs with faked round caps.
    acx, acy = T(ARC_C)
    sw = max(1, round(k * ARC_SW))
    for radius, color, alpha in ARCS:
        col = color[:3] + (alpha,)
        rr = k * radius
        d.arc([acx - rr, acy - rr, acx + rr, acy + rr],
              ARC_START, ARC_END, fill=col, width=sw)
        for ang in (ARC_START, ARC_END):
            a = math.radians(ang)
            ex, ey = acx + rr_mid(rr, sw) * math.cos(a), acy + rr_mid(rr, sw) * math.sin(a)
            cap = sw / 2 - 0.5
            d.ellipse([ex - cap, ey - cap, ex + cap, ey + cap], fill=col)


def rr_mid(rr: float, sw: int) -> float:
    """PIL strokes arcs inward from the bbox — cap centers sit mid-stroke."""
    return rr - sw / 2


def make_icon(size: int = 256, ss: int = 4) -> Image.Image:
    S = size * ss
    k = S / 256.0
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=k * 56, fill=BG)
    draw_mark(img, 0, 0, k)
    return img.resize((size, size), Image.LANCZOS)


def make_logo(w: int = 600, h: int = 240, ss: int = 3) -> Image.Image:
    W, H = w * ss, h * ss
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=ss * 32, fill=BG)
    # Mark on the left (no own plate), vertically centered on the head.
    k = H / 256.0 * 0.92
    draw_mark(img, ss * 24, (H - k * 256) / 2 + k * 6, k)
    # Wordmark.
    try:
        font_big = ImageFont.truetype("C:/Windows/Fonts/bahnschrift.ttf", int(H * 0.30))
        font_small = ImageFont.truetype("C:/Windows/Fonts/bahnschrift.ttf", int(H * 0.085))
    except OSError:
        font_big = ImageFont.truetype("arialbd.ttf", int(H * 0.30))
        font_small = ImageFont.truetype("arial.ttf", int(H * 0.085))
    tx = ss * 24 + k * 230
    ty = H * 0.30
    spacing = H * 0.022
    x = tx
    for ch in "MURDOCK":
        d.text((x, ty), ch, font=font_big, fill=CRIMSON)
        x += d.textlength(ch, font=font_big) + spacing
    d.text((tx + ss * 2, ty + H * 0.36), "S P E A K E R   R E C O G N I T I O N",
           font=font_small, fill=MUTED)
    return img.resize((w, h), Image.LANCZOS)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    make_icon(256).save(root / "icon.png")
    make_icon(64, ss=8).save(root / "murdock" / "ui" / "static" / "favicon.png")
    make_logo().save(root / "logo.png")
    print("wrote icon.png, logo.png, murdock/ui/static/favicon.png")


if __name__ == "__main__":
    main()
