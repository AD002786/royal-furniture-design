"""Download every remote product photo into img/p/ and point products.json at the local copies."""
import json, os, io, time, urllib.request, hashlib
from PIL import Image
P = json.load(open('products.json'))
os.makedirs('img/p', exist_ok=True)
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36'
def get(url):
    host = '/'.join(url.split('/')[:3]) + '/'
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Referer': host, 'Accept': 'image/avif,image/webp,image/*,*/*'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()
ok = bad = 0; failed = []
for p in P:
    new = []
    for i, u in enumerate(p.get('gallery', [])):
        if not u.startswith('http'):
            new.append(u); continue
        name = f"img/p/{p['sku']}_{hashlib.md5(u.encode()).hexdigest()[:8]}.jpg"
        try:
            if not os.path.exists(name):
                im = Image.open(io.BytesIO(get(u))).convert('RGB')
                if max(im.size) > 1200: im.thumbnail((1200, 1200))
                if min(im.size) < 250: raise ValueError('too small')
                im.save(name, quality=82, optimize=True, progressive=True)
                time.sleep(0.15)
            new.append(name); ok += 1
        except Exception as e:
            bad += 1; failed.append(f"{p['sku']} {u} {e}")
    if new:
        p['gallery'] = new; p['img'] = new[0]
json.dump(P, open('products.json', 'w'), indent=1)
open('tools/fetch_failed.txt', 'w').write('\n'.join(failed) + '\n')
print('saved', ok, 'failed', bad)
