import json, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torchvision.ops import nms
LABELS = ["a bed", "a dresser", "a tall chest of drawers", "a nightstand", "a mirror", "a dining table", "a chair",
          "a bench", "a sofa", "a loveseat", "an ottoman", "a cabinet", "a coffee table", "a side table", "a bar stool", "a sectional sofa"]
M = "google/owlv2-base-patch16-ensemble"
proc = Owlv2Processor.from_pretrained(M); model = Owlv2ForObjectDetection.from_pretrained(M).eval()
import os
out = json.load(open('tools/boxes.json')) if os.path.exists('tools/boxes.json') else {}
for f in json.load(open('tools/detect_list.json')):
    im = Image.open(f).convert('RGB'); W, H = im.size; S = max(W, H)
    inp = proc(text=[LABELS], images=im, return_tensors='pt')
    with torch.no_grad(): o = model(**inp)
    sc = o.logits[0].sigmoid(); bx = o.pred_boxes[0]
    best, lab = sc.max(-1)
    m0 = best > 0.08
    cx, cy, bw, bh = bx[:, 0] * S, bx[:, 1] * S, bx[:, 2] * S, bx[:, 3] * S
    xy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
    r = {'boxes': xy[m0], 'scores': best[m0], 'labels': lab[m0]}
    keep = []
    for li in range(len(LABELS)):
        m = r['labels'] == li
        if not m.any(): continue
        b, s = r['boxes'][m], r['scores'][m]
        k = nms(b, s, 0.5)[:6]
        for i in k:
            x0, y0, x1, y1 = [round(float(v)) for v in b[i]]
            keep.append([LABELS[li][2:] if LABELS[li].startswith('a ') else LABELS[li][3:], round(float(s[i]), 3), max(0, x0), max(0, y0), min(W, x1), min(H, y1)])
    out[f] = {'w': W, 'h': H, 'boxes': sorted(keep, key=lambda k: -k[1])}
    print(f, len(keep), flush=True)
json.dump(out, open('tools/boxes.json', 'w'))
