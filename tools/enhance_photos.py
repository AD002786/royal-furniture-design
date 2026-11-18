"""Upscale catalog room photos 4x with Real-ESRGAN, then size to 1800px. Usage: enhance_photos.py SHARD NSHARDS"""
import sys, os, glob, numpy as np, torch
from PIL import Image, ImageEnhance, ImageFilter
from spandrel import ModelLoader
shard, n = int(sys.argv[1]), int(sys.argv[2])
torch.set_num_threads(os.cpu_count())
model = ModelLoader().load_from_file('RealESRGAN_x4plus.pth').model.eval()
@torch.inference_mode()
def up(im, tile=320, pad=12):
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)[None]
    _, _, H, W = x.shape; out = torch.zeros(1, 3, H * 4, W * 4)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ya, xa, yb, xb = max(y0 - pad, 0), max(x0 - pad, 0), min(y1 + pad, H), min(x1 + pad, W)
            r = model(x[:, :, ya:yb, xa:xb]).clamp(0, 1)
            out[:, :, y0*4:y1*4, x0*4:x1*4] = r[:, :, (y0-ya)*4:(y0-ya)*4+(y1-y0)*4, (x0-xa)*4:(x0-xa)*4+(x1-x0)*4]
    return Image.fromarray((out[0].permute(1, 2, 0).numpy() * 255).round().astype('uint8'))
os.makedirs('out', exist_ok=True)
files = sorted(glob.glob('tools/src/*.png'))[shard::n]
for f in files:
    im = up(Image.open(f).convert('RGB'))
    im.thumbnail((1800, 1800), Image.LANCZOS)
    im = ImageEnhance.Contrast(im).enhance(1.04); im = ImageEnhance.Color(im).enhance(1.05)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=40, threshold=2))
    im.save('out/' + os.path.basename(f)[:-4] + '.jpg', quality=86, optimize=True, progressive=True)
    print(f, im.size, flush=True)
