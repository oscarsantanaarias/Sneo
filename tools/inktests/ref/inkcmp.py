import sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from tokenizers import Tokenizer
raw = open(sys.argv[1], 'rb').read()
count = int(np.frombuffer(raw[:4], np.int32)[0])
ids = np.frombuffer(raw[4:4 + 4 * count], np.int32).tolist()
ours = np.frombuffer(raw[4 + 4 * count:], np.float32)
tok = Tokenizer.from_file(sys.argv[2])
text = tok.decode(ids, skip_special_tokens=False)
print('ids', ','.join(map(str, ids)))
print('text', repr(text))
again = tok.encode(text, add_special_tokens=False).ids
print('hf tokenization same:', again == ids)
if len(sys.argv) > 3:
    ref = np.load(sys.argv[3]).astype(np.float32)
    print("ref shape", ref.shape)
    ref = ref.reshape(-1, ref.shape[-1])[-1]
    keep = np.isfinite(ref) & np.isfinite(ours[:ref.size])
    print("finite", int(keep.sum()), "of", ref.size)
    ref = np.where(keep, ref, -1e9)
    ours = np.where(keep, ours[:ref.size], -1e9)
    o, r = ours[:ref.size], ref
    print("corr", float(np.corrcoef(o[keep], r[keep])[0, 1]), "maxdiff", float(np.abs(o[keep] - r[keep]).max()))
    for name, v in (('ours', o), ('ref', r)):
        top = np.argsort(-v)[:6]
        print(name, [(tok.decode([int(i)]), round(float(v[i]), 3)) for i in top])
    def lse(v):
        m = v.max()
        return m + np.log(np.exp(v - m).sum())
    p = np.exp(r - lse(r))
    print('kl ref||ours', float((p * ((r - lse(r)) - (o - lse(o)))).sum()))
