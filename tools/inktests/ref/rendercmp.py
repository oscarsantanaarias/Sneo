import sys, json, subprocess, numpy as np
from tokenizers import Tokenizer
sys.stdout.reconfigure(encoding="utf-8")
raw = open(sys.argv[1], "rb").read()
n = int(np.frombuffer(raw[:4], np.int32)[0])
ours = np.frombuffer(raw[4:4 + 4 * n], np.int32).tolist()
tok = Tokenizer.from_file(sys.argv[2])
official = open(sys.argv[3], encoding="utf-8").read().split("\n", 1)[1].rstrip("\n")
hf = tok.encode(official, add_special_tokens=False).ids
print("same", ours == hf, len(ours), len(hf))
for i in range(min(len(ours), len(hf)) + 1):
    if i >= len(ours) or i >= len(hf) or ours[i] != hf[i]:
        print("first diff at", i)
        print("ours", repr(tok.decode(ours[max(0, i - 4):i + 12], skip_special_tokens=False)))
        print("hf  ", repr(tok.decode(hf[max(0, i - 4):i + 12], skip_special_tokens=False)))
        break
