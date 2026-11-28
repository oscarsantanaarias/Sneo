import sys, numpy as np
from tokenizers import Tokenizer
sys.stdout.reconfigure(encoding="utf-8")
raw = open(sys.argv[1], "rb").read()
n = int(np.frombuffer(raw[:4], np.int32)[0])
ours = np.frombuffer(raw[4:4 + 4 * n], np.int32).tolist()
tok = Tokenizer.from_file(sys.argv[2])
text = tok.decode(ours, skip_special_tokens=False)
print(repr(text))
hf = tok.encode(text, add_special_tokens=False).ids
print("same", ours == hf, len(ours), len(hf))
for i in range(max(len(ours), len(hf))):
    a = ours[i] if i < len(ours) else None
    b = hf[i] if i < len(hf) else None
    if a != b:
        print("first diff at", i)
        print("ours", [(x, tok.decode([x])) for x in ours[i:i + 8]])
        print("hf  ", [(x, tok.decode([x])) for x in hf[i:i + 8]])
        break
