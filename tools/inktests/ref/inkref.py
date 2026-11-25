import sys, os, math, time, argparse
import numpy as np
import gguf
from gguf import GGUFReader
from gguf.quants import dequantize

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=r'F:\AI Projects\Strata\Models\Inkling-2-Mini\Inkling-2-Mini-UD-Q4_K_XL.gguf')
parser.add_argument('--ids', default='')
parser.add_argument('--text', default='The capital of France is')
parser.add_argument('--tokenizer', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'tokenizer.json'))
parser.add_argument('--out', default='')
args = parser.parse_args()

reader = GGUFReader(args.model)
tensors = {t.name: t for t in reader.tensors}
meta = {}
for field in reader.fields.values():
    try:
        value = field.contents()
    except Exception:
        continue
    meta[field.name] = value

def get(name):
    t = tensors[name]
    shape = [int(x) for x in t.shape][::-1]
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
        data = dequantize(np.asarray(t.data), t.tensor_type) if t.tensor_type == gguf.GGMLQuantizationType.BF16 else np.asarray(t.data, dtype=np.float32)
        return np.asarray(data, dtype=np.float32).reshape(shape)
    return np.asarray(dequantize(np.asarray(t.data), t.tensor_type), dtype=np.float32).reshape(shape)

def expert(name, e):
    t = tensors[name]
    raw = np.asarray(t.data)
    rows = int(t.shape[1])
    cols = int(t.shape[0])
    return np.asarray(dequantize(raw[e], t.tensor_type), dtype=np.float32).reshape(rows, cols)

A = 'inkling2_loop.'
hidden = 2304
layers = 24
heads = 24
kv_heads = 8
head_dim = 128
eps = 1e-6
loop_start, loop_end, passes = 6, 18, 3
block = 6
window = 1024
extent = 1024
qk_clip = float(meta[A + 'attention.qk_clip'])
rel_clip = float(meta[A + 'attention.relative_clip'])
floor_n = float(meta[A + 'attention.log_scale_floor'])
alpha = float(meta[A + 'attention.log_scale_alpha'])
softcap_logits = float(meta[A + 'final_logit_softcapping'])
gate_cap = float(meta[A + 'activation.situ_beta'])
up_cap = float(meta[A + 'activation.situ_linear_beta'])
route_scale = float(meta[A + 'expert_weights_scale'])
pattern = list(meta[A + 'attention.sliding_window_pattern'])

def rms(x, w=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y * w if w is not None else y

def l2clip(x, threshold):
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x * np.minimum(threshold / (norm + 1e-6), 1.0)

def log_tau(pos):
    return 1.0 + alpha * np.log(np.maximum((pos + 1.0) / floor_n, 1.0))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def short_conv(x, w):
    taps = w.shape[1]
    pad = np.concatenate([np.zeros((taps - 1, x.shape[1]), np.float32), x], axis=0)
    out = np.zeros_like(x)
    for j in range(taps):
        out += pad[j:j + x.shape[0]] * w[:, j][None, :]
    return (out + x) / math.sqrt(taps + 1)

def dynamic_conv(x, dyn_in, static, proj, kv):
    taps = static.shape[1]
    seq = x.shape[0]
    dyn = (dyn_in @ proj.T).reshape(seq, kv, taps)
    per = x.shape[1] // kv
    pad = np.concatenate([np.zeros((taps - 1, x.shape[1]), np.float32), x], axis=0)
    out = np.zeros_like(x)
    for lag in range(taps):
        back = pad[taps - 1 - lag: taps - 1 - lag + seq]
        weight = static[:, taps - 1 - lag][None, :] + np.repeat(dyn[:, :, lag], per, axis=1)
        out += back * weight
    return (out + x) / math.sqrt(2 * taps + 1)

def block_res(states, partial, queries):
    stacked = np.stack(states + [partial], axis=1)
    stacked = rms(stacked)
    outs = []
    for q in queries:
        logits = np.einsum('skd,d->sk', stacked, q) / math.sqrt(hidden)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        outs.append(np.einsum('sk,skd->sd', p, stacked))
    return outs

cache = {}
def W(name):
    if name not in cache:
        cache[name] = get(name)
    return cache[name]

def message_gate(l, kind, h, m):
    p = f'blk.{l}.loop_{kind}_ctrl_'
    g = np.concatenate([rms(h), rms(m)], axis=-1) @ W(p + 'down.weight').T
    forget = np.tanh(g @ W(p + 'forget.weight').T)
    gain = np.tanh(g @ W(p + 'gain.weight').T)
    return (1 - forget) * h + gain * m

def attention(l, hq, hk, hv, positions):
    p = f'blk.{l}.'
    full = pattern[l] == 0
    seq = hq.shape[0]
    q = (hq @ W(p + 'attn_q.weight').T).reshape(seq, heads, head_dim)
    q = l2clip(q * W(p + 'attn_q_clip.weight'), qk_clip)
    if full:
        q = q * log_tau(positions)[:, None, None]
    k = hk @ W(p + 'attn_k.weight').T
    v = hv @ W(p + 'attn_v.weight').T
    if full:
        k = dynamic_conv(k, hk, W(p + 'attn_k_conv.weight'), W(p + 'attn_k_conv_proj.weight'), kv_heads)
        v = dynamic_conv(v, hv, W(p + 'attn_v_conv.weight'), W(p + 'attn_v_conv_proj.weight'), kv_heads)
    else:
        k = short_conv(k, W(p + 'attn_k_conv.weight'))
        v = short_conv(v, W(p + 'attn_v_conv.weight'))
    k = l2clip(k.reshape(seq, kv_heads, head_dim), qk_clip)
    v = rms(v.reshape(seq, kv_heads, head_dim))
    rs = (hq @ W(p + 'attn_rel.weight').T).reshape(seq, heads, 16)
    rs = l2clip(rs * W(p + 'attn_rel_clip.weight'), rel_clip)
    if full:
        rs = rs * log_tau(positions)[:, None, None]
    bank = W(p + 'attn_rel_bias.weight')
    rel = np.einsum('shi,ri->shr', rs, bank)
    sinks = hq @ W(p + 'attn_sink_proj.weight').T
    out = np.zeros((seq, heads, head_dim), np.float32)
    for h in range(heads):
        g = h // (heads // kv_heads)
        scores = (q[:, h, :] @ k[:, g, :].T) / math.sqrt(head_dim)
        dist = (positions[:, None] - positions[None, :]).astype(np.int64)
        bias = np.where((dist >= 0) & (dist < extent), np.take_along_axis(rel[:, h, :], np.clip(dist, 0, extent - 1), axis=1), 0.0)
        scores = scores + bias
        allowed = dist >= 0
        if not full:
            allowed &= dist < window
        scores = np.where(allowed, scores, -np.inf)
        top = np.maximum(scores.max(axis=1), sinks[:, h])
        e = np.exp(scores - top[:, None])
        total = e.sum(axis=1) + np.exp(sinks[:, h] - top)
        out[:, h, :] = (e @ v[:, g, :]) / total[:, None]
    gate = sigmoid(hq @ W(p + 'attn_gate.weight').T)
    out = (out.reshape(seq, -1) * gate) @ W(p + 'attn_output.weight').T
    return short_conv(out, W(p + 'attn_post_conv.weight'))

def moe(l, x):
    p = f'blk.{l}.'
    logits = x @ W(p + 'ffn_gate_inp.weight').T
    scores = sigmoid(logits)
    choice = scores + W(p + 'exp_probs_b.bias')
    top = np.argsort(-choice, axis=1)[:, :8]
    weights = np.take_along_axis(scores, top, axis=1)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-20) * route_scale
    gu_scale = W(p + 'ffn_gate_up_exps.scale') if p + 'ffn_gate_up_exps.scale' in tensors else None
    d_scale = W(p + 'ffn_down_exps.scale') if p + 'ffn_down_exps.scale' in tensors else None
    out = np.zeros_like(x)
    for e in np.unique(top):
        rows, slots = np.where(top == e)
        gu = x[rows] @ expert(p + 'ffn_gate_up_exps.weight', e).T
        if gu_scale is not None:
            gu = gu * gu_scale[e]
        gate, up = gu[:, :gu.shape[1] // 2], gu[:, gu.shape[1] // 2:]
        act = np.tanh(gate / gate_cap) * gate_cap * sigmoid(gate) * np.tanh(up / up_cap) * up_cap
        down = act @ expert(p + 'ffn_down_exps.weight', e).T
        if d_scale is not None:
            down = down * d_scale[e]
        out[rows] += down * weights[rows, slots][:, None]
    return short_conv(out, W(p + 'ffn_post_conv.weight'))

def layer(l, h, states, positions, messages=None):
    p = f'blk.{l}.'
    full = pattern[l] == 0
    residual_in = h
    if full:
        if l > 0:
            hq, hk, hv = block_res(states, h, list(W(p + 'attn_res_query.weight')))
        else:
            hq = hk = hv = h
        if messages is not None:
            hq = message_gate(l, 'q', hq, messages[0])
            hk = message_gate(l, 'k', hk, messages[1])
            hv = message_gate(l, 'v', hv, messages[2])
        hq = rms(hq, W(p + 'attn_q_inp_norm.weight'))
        hk = rms(hk, W(p + 'attn_k_inp_norm.weight'))
        hv = rms(hv, W(p + 'attn_v_inp_norm.weight'))
    else:
        x = block_res(states, h, [W(p + 'attn_res_query.weight')])[0] if l > 0 else h
        norm = W(p + 'attn_norm.weight')
        if messages is None:
            hq = hk = hv = rms(x, norm)
        else:
            hq = rms(message_gate(l, 'q', x, messages[0]), norm)
            hk = rms(message_gate(l, 'k', x, messages[1]), norm)
            hv = rms(message_gate(l, 'v', x, messages[2]), norm)
    attn = attention(l, hq, hk, hv, positions)
    inner = list(states)
    if l % block == 0:
        inner.append(residual_in)
        residual = attn
    else:
        residual = residual_in + attn
    m = block_res(inner, residual, [W(p + 'ffn_res_query.weight')])[0]
    if messages is not None:
        m = message_gate(l, 'ffn', m, messages[3])
    return residual + moe(l, rms(m, W(p + 'ffn_norm.weight')))

if args.ids:
    ids = [int(x) for x in args.ids.split(',')]
    pieces = None
else:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    enc = tok.encode(args.text, add_special_tokens=False)
    ids = enc.ids
    pieces = enc.tokens
print('ids:', ','.join(map(str, ids)))
began = time.time()
positions = np.arange(len(ids), dtype=np.float32)
embed = tensors['token_embd.weight']
rows = np.asarray(embed.data)
h = np.stack([np.asarray(dequantize(rows[i:i + 1], embed.tensor_type), dtype=np.float32).reshape(-1) for i in ids])
h = rms(h, W('token_embd_norm.weight'))
states = []
for l in range(loop_start):
    nxt = layer(l, h, states, positions)
    if l % block == 0:
        states = states + [h]
    h = nxt
pre_h, pre_states = h, list(states)
pre_blocks = len(pre_states)
for l in range(loop_start, loop_end):
    nxt = layer(l, h, states, positions)
    if l % block == 0:
        states = states + [h]
    h = nxt
for passno in range(1, passes):
    prev_states = states[pre_blocks + 1:]
    prev_h = h
    h, states = pre_h, list(pre_states)
    for j, l in enumerate(range(loop_start, loop_end)):
        queries = W(f'blk.{l}.loop_msg_query.weight')
        msgs = block_res(prev_states, prev_h, list(queries))
        nxt = layer(l, h, states, positions, msgs)
        if l % block == 0:
            states = states + [h]
        h = nxt
for l in range(loop_end, layers):
    nxt = layer(l, h, states, positions)
    if l % block == 0:
        states = states + [h]
    h = nxt
h = block_res(states, h, [W('output_res_query.weight')])[0]
h = rms(h, W('output_norm.weight'))
head = tensors['output.weight']
head_rows = np.asarray(head.data)
logits = np.zeros((len(ids), int(head.shape[1])), np.float32)
step = 8192
for start in range(0, logits.shape[1], step):
    part = np.asarray(dequantize(head_rows[start:start + step], head.tensor_type), dtype=np.float32)
    logits[:, start:start + part.shape[0]] = h @ part.T
logits = np.tanh(logits / softcap_logits) * softcap_logits
logits[:, 200058:] = -np.inf
if args.out:
    np.save(args.out, logits)
try:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    name = lambda i: repr(tok.decode([int(i)]))
except Exception:
    name = lambda i: str(i)
nll = []
for t in range(len(ids)):
    order = np.argsort(-logits[t])[:6]
    print(f'pos {t} id {ids[t]} -> ' + '  '.join(f'{name(i)}:{logits[t, i]:.3f}' for i in order))
    if t + 1 < len(ids):
        z = logits[t] - logits[t].max()
        nll.append(-(z[ids[t + 1]] - math.log(np.exp(z).sum())))
if nll:
    print(f'mean nll {np.mean(nll):.4f}')
print(f'elapsed {time.time() - began:.1f} s')
