import sys, re
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
layers = {}
for t in r.tensors:
    m = re.match(r'blk\.(\d+)\.(.+)', t.name)
    if not m:
        continue
    l, name = int(m.group(1)), m.group(2)
    d = layers.setdefault(l, {'attn': 0, 'exp': 0, 'other': 0})
    b = int(t.n_bytes)
    if '_exps' in name:
        d['exp'] += b
    elif name.startswith('attn') or name.startswith('post_attn'):
        d['attn'] += b
    else:
        d['other'] += b
runs = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
attn = sum(layers[l]['attn'] for l in runs)
exp = sum(layers[l]['exp'] * 8 / 128 for l in runs)
other = sum(layers[l]['other'] for l in runs)
print('per token over 48 runs: attention %.0f MB, 8 experts %.0f MB, other %.0f MB, total %.0f MB' % (attn / 2**20, exp / 2**20, other / 2**20, (attn + exp + other) / 2**20))
for gbps in (936, 750):
    print('at %d GB/s: attention %.2f ms, experts %.2f ms, total %.2f ms' % (gbps, attn / gbps / 1e6, exp / gbps / 1e6, (attn + exp + other) / gbps / 1e6))
