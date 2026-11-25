import json, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torchvision.ops import nms
LABELS = ["a bed", "a dresser", "a tall chest of drawers", "a nightstand", "a mirror", "a dining table", "a chair",
          "a bench", "a sofa", "a loveseat", "an ottoman", "a cabinet", "a coffee table", "a side table", "a bar stool", "a sectional sofa"]
M = "google/owlv2-base-patch16-ensemble"
proc = Owlv2Processor.from_pretrained(M); model = Owlv2ForObjectDetection.from_pretrained(M).eval()
out = {}
for f in json.load(open('tools/detect_list.json')):
    im = Image.open(f).convert('RGB'); W, H = im.size; S = max(W, H)
    inp = proc(text=[LABELS], images=im, return_tensors='pt')
    with torch.no_grad(): o = model(**inp)
    r = proc.post_process_object_detection(o, threshold=0.08, target_sizes=torch.tensor([[S, S]]))[0]
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
