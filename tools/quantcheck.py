import sys
import struct
import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

names = {0: 'F32', 1: 'F16', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS', 30: 'BF16', 42: 'Q2_0'}
sizes = {0: (1, 4), 1: (1, 2), 12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 30: (1, 2), 42: (64, 18)}


def ternary(raw):
    blocks = raw.reshape(-1, 18)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    codes = (blocks[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
    return ((codes.reshape(-1, 64).astype(np.float32) - 1) * d).reshape(-1)


data = open(sys.argv[1], 'rb').read()
at = 0
stats = {}
while at < len(data):
    kind, cols, length = struct.unpack_from('<III', data, at)
    at += 12
    name = data[at:at + length].decode()
    at += length
    row, = struct.unpack_from('<Q', data, at)
    at += 8
    per, size = sizes[kind]
    count = cols // per * size
    raw = np.frombuffer(data, dtype=np.uint8, count=count, offset=at).copy()
    at += count
    mine = np.frombuffer(data, dtype=np.float32, count=cols, offset=at)
    at += cols * 4

    if kind == 42:
        ref = ternary(raw)
    else:
        ref = np.asarray(dequantize(raw, GGMLQuantizationType(kind)), dtype=np.float32).reshape(-1)

    diff = np.abs(ref.astype(np.float64) - mine.astype(np.float64))
    worst = float(diff.max())
    scale = float(np.abs(ref).max())
    exact = int(np.sum(ref == mine))
    s = stats.setdefault(kind, {'rows': 0, 'elements': 0, 'exact': 0, 'worst': 0.0, 'relative': 0.0, 'where': ''})
    s['rows'] += 1
    s['elements'] += cols
    s['exact'] += exact
    if worst > s['worst']:
        s['worst'] = worst
        s['where'] = '%s row %d' % (name, row)
    if scale > 0:
        s['relative'] = max(s['relative'], worst / scale)

print('%-4s %-8s %5s %10s %10s %12s %12s  %s' % ('id', 'type', 'rows', 'elements', 'exact %', 'max abs err', 'max rel err', 'reference'))
for kind in sorted(stats):
    s = stats[kind]
    print('%-4d %-8s %5d %10d %9.4f%% %12.3e %12.3e  %s' % (kind, names[kind], s['rows'], s['elements'], 100.0 * s['exact'] / s['elements'], s['worst'], s['relative'], 'spec formula (numpy)' if kind == 42 else 'gguf.quants.dequantize'))
