"""Original, explicitly simulated grayscale scenes for the offline UI demo."""

import random

from PIL import Image, ImageDraw


def _scene(width, height, seed):
    rng = random.Random(seed)
    image = Image.new('L', (width, height), 49)
    draw = ImageDraw.Draw(image)
    horizon = int(height * .55)
    draw.rectangle((0, 0, width, horizon), fill=105)
    draw.polygon([(0, horizon), (width, horizon), (width, height), (0, height)], fill=63)
    for y in (horizon + 38, horizon + 95, horizon + 170, height - 15):
        draw.line((0, y, width, y), fill=94, width=2)
    for x in range(60, width, 145):
        draw.rectangle((x, horizon - 110, x + 82, horizon - 8), fill=154, outline=206,
                       width=2)
        draw.ellipse((x + 29, horizon - 71, x + 52, horizon - 48), fill=72)
    for x in range(90, width, 175):
        draw.polygon([(x, height - 52), (x + 65, height - 76),
                      (x + 115, height - 44), (x + 51, height - 19)], fill=145)
        draw.polygon([(x, height - 52), (x + 51, height - 19),
                      (x + 51, height - 92), (x, height - 121)], fill=104)
    for _ in range(650):
        x, y = rng.randrange(width), rng.randrange(28, height)
        shade = rng.randrange(70, 190)
        draw.point((x, y), fill=shade)
    return image


def _mark(image, label):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 27), fill=31)
    draw.text((12, 8), f'SIMULATED GRAYSCALE SCENE  |  {label}', fill=235)
    return image


def make_demo_frames():
    """Return five bounded PIL L images; the front pair shares an overlap."""
    front = _scene(1080, 480, seed=32)
    return {
        'frontleft_fisheye_image': _mark(front.crop((0, 0, 720, 480)), 'FRONT LEFT'),
        'frontright_fisheye_image': _mark(front.crop((360, 0, 1080, 480)), 'FRONT RIGHT'),
        'left_fisheye_image': _mark(_scene(720, 480, seed=47), 'LEFT'),
        'right_fisheye_image': _mark(_scene(720, 480, seed=61), 'RIGHT'),
        'back_fisheye_image': _mark(_scene(720, 480, seed=83), 'BACK'),
    }


def make_demo_panorama():
    """Known-overlap composite for UI inspection, not an SDK camera stitch."""
    return _mark(_scene(1080, 480, seed=32), 'SIMULATED FRONT COMPOSITE')
