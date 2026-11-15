import json
import random
import struct
import sys
import time
from pathlib import Path

import regex
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE

SPLIT = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def read_meta(path):
    scalars = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with open(path, "rb") as f:
        def take(kind):
            size = struct.calcsize("<" + kind)
            return struct.unpack("<" + kind, f.read(size))[0]

        def text():
            return f.read(take("Q")).decode("utf-8")

        def value(kind):
            if kind == 8:
                return text()
            if kind == 9:
                inner = take("I")
                count = take("Q")
                return [value(inner) for _ in range(count)]
            return take(scalars[kind])

        if f.read(4) != b"GGUF":
            raise SystemExit("not a gguf file: " + path)
        take("I")
        take("Q")
        count = take("Q")
        meta = {}
        for _ in range(count):
            key = text()
            meta[key] = value(take("I"))
        return meta


def build_tokenizer(tokens, merges, types, kinds):
    pairs = [tuple(m.split(" ", 1)) for m in merges]
    model = BPE(vocab={t: i for i, t in enumerate(tokens)}, merges=pairs, fuse_unk=False, byte_fallback=False)
    tok = Tokenizer(model)
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(tokens[i], special=True, normalized=False) for i, t in enumerate(types) if t in kinds])
    for i, t in enumerate(types):
        if t in kinds and tok.token_to_id(tokens[i]) != i:
            raise SystemExit("special token id moved: " + tokens[i])
    return tok


FIXED = [
    "Hello world",
    "Hello, world! How are you today?",
    "The quick brown fox jumps over the lazy dog. It's 5 o'clock, isn't it? We'll see; they'd've known.",
    "I'M HERE, YOU'RE THERE, THEY'VE GONE, HE'LL STAY, SHE'D GO, IT'S OK, DON'T",
    "x'll y'LL z'Ll w'lL 'S 's 't 'T 're 'RE 'Re 've 'VE 'm 'M 'd 'D 'x ''s '''",
    "El niño comió piña en la mañana. ¿Dónde está el pingüino? ¡Qué año tan extraño!",
    "Canción, corazón, acción, educación, pequeño, señor, Ñandú, ÁÉÍÓÚ áéíóú üÜ ç Ç",
    "Mañana, el señor Muñoz llegó tarde: ¡qué vergüenza! Él dijo «ya voy» — y se fue…",
    "def main():\n\tfor i in range(10):\n\t\tprint(i)\n\n    return  0\n",
    "int main() {\n    int x = 12345;\n    double pi = 3.14;\n    return x;\n}\n",
    "if(a==b){\r\n    return;\r\n}\r\n",
    "\t\t\tindented with tabs\n        indented with spaces\n  \t mixed\t  \n",
    "for(size_t i = 0; i < n; i++){\n        out += table[i] ^ 0xFF;\n    }\n",
    "    \n\n\n    \n\t\n",
    "12345 3.14 -0.001 1e10 1,000,000 0x1F 2026-11-07 99% $12.50 €3,99",
    "1234567890 00000 123abc abc123 a1b2c3 3.14159265358979",
    "Emojis: 😀😃😄 👨‍👩‍👧‍👦 🇪🇸 👍🏽 ❤️ ✨🔥 🤖",
    "我们今天去公园散步，天气非常好！这是一个测试。",
    "こんにちは、世界。今日はいい天気ですね。カタカナとひらがな、漢字。",
    "안녕하세요 세계, 반갑습니다.",
    "Здравствуй, мир! Ελληνικά γράμματα.",
    "مرحبا بالعالم ١٢٣ وَعَلَيْكُم",
    "हिन्दी में लिखा गया वाक्य १२३",
    "ภาษาไทยกินข้าวหรือยัง",
    "Combining: é à ñ ȫ ́alone x⃝",
    "Math: ∑_{i=1}^{n} x² ≤ ∞, α+β=γ, ½ ¾ Ⅻ ①② ³√8",
    "URLs: https://example.com/path?q=1&r=2#frag user@mail.com",
    "JSON: {\"key\": [1, 2, 3], \"nested\": {\"a\": null, \"b\": true}}",
    "  leading spaces\t\ttabs   \n\n\n  trailing   \n \r\n\r\n\t  nbsp　ideographic em",
    "a  b   c    d\n\ne\n \nf \n g",
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\nhmm\n</think>\n\nHi!<|im_end|>",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\n¿Hola?<|im_end|>\n",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nMadrid\n</parameter>\n</function>\n</tool_call>",
    "<tool_response>\n{\"ok\": true}\n</tool_response>",
    "almost special <|im_star and <think without close <|endoftext|><|endoftext|>",
    "nested <<|im_start|>> and <|im_end|><|im_end|> back to back<think></think>",
    "<|vision_start|><|image_pad|><|vision_end|> [PAD248100] <|fim_prefix|>x<|fim_suffix|>",
    "...!!!???", "—–-", "\x00\x01\x1f\x7f", "\x1c\x1d\x1e\x1f\x0b\x0c", "\u0085  ",
    "", " ", "  ", "\n", "\r\n", "\t", "a", "'", "''''", "'s's'S",
    "word " * 300,
    "a" * 5000,
    " " * 3000,
    "\n" * 1000,
    "中文测试" * 1500,
    "😀" * 800,
    "e" + "́" * 300,
    "ab" * 2000 + "!",
]


def random_texts(count, seed):
    rng = random.Random(seed)
    pools = [
        (8, [chr(c) for c in range(0x20, 0x7F)]),
        (5, list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")),
        (3, list("0123456789")),
        (5, [" ", " ", " ", "\t", "\n", "\r", "\r\n", "\x0b", "\x0c", "\x1c", "\x85", "\xa0", " ", " ", " ", " ", " ", " ", "　", "​", "﻿"]),
        (3, ["'", "'s", "'t", "'re", "'ve", "'ll", "'d", "'m", "'S", "'LL", "'Re", "'x"]),
        (3, [chr(c) for c in range(0xC0, 0x250)]),
        (3, [chr(c) for c in range(0x300, 0x370)]),
        (2, [chr(c) for c in range(0x391, 0x3CA)] + [chr(c) for c in range(0x410, 0x450)]),
        (2, [chr(c) for c in range(0x621, 0x64B)] + [chr(c) for c in range(0x64B, 0x653)] + [chr(c) for c in range(0x660, 0x66A)]),
        (2, [chr(c) for c in range(0x905, 0x93A)] + [chr(c) for c in range(0x93E, 0x94E)] + [chr(c) for c in range(0x966, 0x970)]),
        (2, [chr(c) for c in range(0xE01, 0xE2F)] + [chr(c) for c in range(0xE31, 0xE3B)]),
        (4, [chr(c) for c in range(0x4E00, 0x9FA6)]),
        (2, [chr(c) for c in range(0x3041, 0x3097)] + [chr(c) for c in range(0x30A1, 0x30FB)] + ["、", "。", "ー"]),
        (1, [chr(c) for c in range(0xAC00, 0xD7A4)]),
        (2, ["²", "³", "¹", "¼", "½", "¾"] + [chr(c) for c in range(0x2160, 0x2183)] + [chr(c) for c in range(0x2460, 0x249C)] + [chr(c) for c in range(0xFF10, 0xFF1A)]),
        (2, [chr(c) for c in range(0x2190, 0x2200)] + [chr(c) for c in range(0x2200, 0x2300)] + [chr(c) for c in range(0x2500, 0x2580)] + [chr(c) for c in range(0x20A0, 0x20C0)]),
        (2, [chr(c) for c in range(0x1F300, 0x1F650)] + [chr(c) for c in range(0x1F900, 0x1FA00)] + ["‍", "️"] + [chr(c) for c in range(0x1F3FB, 0x1F400)] + [chr(c) for c in range(0x1F1E6, 0x1F200)]),
        (1, [chr(c) for c in range(0xFF01, 0xFF5F)]),
        (1, [chr(c) for c in range(0x00, 0x20)] + ["\x7f"]),
        (1, ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<|endoftext|>", "<|im_", "<tool", "|>"]),
    ]
    weights = [w for w, _ in pools]
    texts = []
    for _ in range(count):
        parts = []
        length = rng.randint(1, 48)
        while len(parts) < length:
            pool = rng.choices(pools, weights)[0][1]
            for _ in range(rng.randint(1, 6)):
                parts.append(rng.choice(pool))
        texts.append("".join(parts))
    return texts


def file_chunks(root):
    names = ["master.cpp", "worker.cpp", "MiEstilo.MD", "src/gguf.h", "src/json.h", "src/net.h", "src/pool.h", "src/hardware.h"]
    chunks = []
    for name in names:
        path = root / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for at in range(0, len(text), 2500):
            chunks.append(text[at:at + 2500])
    return chunks


TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city — returns °C",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name, e.g. Málaga"},
            "days": {"type": "integer", "minimum": 1, "maximum": 7},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "run_code",
        "description": "Run a snippet\nwith \"quotes\", a tab\there and \\ backslash <tags> & 'single'",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"},
            "timeout": {"type": "number", "default": 2.5},
            "flags": {"type": "array", "items": {"type": "string"}},
            "env": {"type": "object"},
            "verbose": {"type": "boolean"}},
            "required": ["code"]}}},
]

CHATS = [
    {"messages": [{"role": "user", "content": "Hola, ¿qué tal?"}]},
    {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 = 4.", "reasoning_content": "\nSimple arithmetic.\n"},
        {"role": "user", "content": "And 3+3?"}],
     "options": {"reasoning_effort": "low"}},
    {"messages": [
        {"role": "system", "content": "You are helpful. Use tools when needed."},
        {"role": "user", "content": "What's the weather in Madrid, and run print(1)?"},
        {"role": "assistant", "content": "Let me check both.", "reasoning_content": "Need two tools.",
         "tool_calls": [
             {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Madrid\", \"days\": 3}"}},
             {"id": "call_2", "type": "function", "function": {"name": "run_code", "arguments": {"code": "print(1)\nprint('ñ')", "timeout": 2.5, "flags": ["-u", "-v"], "env": {"A": "1", "B": None, "C": True}, "verbose": False}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21, \"sky\": \"clear\"}"},
        {"role": "tool", "tool_call_id": "call_2", "content": "1\nñ\n"},
        {"role": "assistant", "content": "It is 21 °C and clear in Madrid. The code printed 1 and ñ.", "reasoning_content": "Summarize results."},
        {"role": "user", "content": "Thanks!"}],
     "tools": TOOLS, "options": {"reasoning_effort": "medium"}},
    {"messages": [
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the chicken cross the road?"},
        {"role": "user", "content": "Why?"}],
     "options": {"enable_thinking": False}},
    {"messages": [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer", "reasoning_content": "first thoughts"},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "call the tool",
         "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": "sunny"}],
     "tools": TOOLS, "options": {"preserve_thinking": False}},
    {"messages": [
        {"role": "developer", "content": "  Developer rules.  "},
        {"role": "system", "content": [{"type": "text", "text": "System part one. "}, {"type": "text", "text": "Part two.\n"}]},
        {"role": "user", "content": [{"type": "text", "text": "  Look at this: "}, {"type": "image_url", "image_url": {"url": "data:x"}}, {"type": "text", "text": " what is it?  "}]},
        {"role": "assistant", "content": None,
         "tool_calls": [{"type": "function", "function": {"name": "run_code", "arguments": {"code": "identify()"}}},
                        {"type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
        {"role": "tool", "content": "a cat"},
        {"role": "tool", "content": [{"type": "text", "text": "cloudy"}]}],
     "options": {"reasoning_effort": "high", "add_generation_prompt": False}},
    {"messages": [
        {"role": "user", "content": "Start"},
        {"role": "assistant", "content": "Calling.", "reasoning_content": "kept or not?",
         "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Oslo", "days": 2}}}]},
        {"role": "user", "content": "<tool_response>\nrain\n</tool_response>"},
        {"role": "assistant", "content": "Rain in Oslo.", "reasoning_content": "after tool"}],
     "tools": TOOLS, "options": {"preserve_thinking": False, "enable_thinking": True}},
    {"messages": [
        {"role": "system", "content": "   "},
        {"role": "user", "content": "  spaced　\n"}],
     "tools": TOOLS, "options": {"enable_thinking": False}},
    {"messages": [
        {"role": "system", "content": "Only system."},
        {"role": "user", "content": "Hi"}],
     "options": {"enable_thinking": True, "reasoning_effort": "xhigh"}},
    {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "late system"}],
     "expect_error": True},
    {"messages": [{"role": "user", "content": "Hi"}], "options": {"reasoning_effort": "extreme"}, "expect_error": True},
    {"messages": [{"role": "narrator", "content": "Hi"}], "expect_error": True},
]


def make_template(source):
    def fail(message):
        raise ValueError(message)

    def as_json(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, sort_keys=sort_keys, separators=separators, indent=indent, ensure_ascii=ensure_ascii)

    jinja = ImmutableSandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"], lstrip_blocks=True, trim_blocks=True)
    jinja.filters.update(tojson=as_json)
    jinja.globals.update(raise_exception=fail)
    return jinja.from_string(source)


def render_chat(template, chat):
    messages = json.loads(json.dumps(chat["messages"]))
    for message in messages:
        for call in message.get("tool_calls") or []:
            target = call.get("function", call)
            if isinstance(target.get("arguments"), str):
                target["arguments"] = json.loads(target["arguments"])
    options = dict(chat.get("options", {}))
    generation = options.pop("add_generation_prompt", True)
    return template.render(messages=messages, tools=chat.get("tools"), add_generation_prompt=generation, **options)


def main():
    model_path = sys.argv[1]
    out_path = sys.argv[2]
    root = Path(__file__).resolve().parent.parent
    started = time.time()
    meta = read_meta(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    merges = meta["tokenizer.ggml.merges"]
    types = meta["tokenizer.ggml.token_type"]
    print("reference: %d tokens, %d merges, pre=%s, read in %.1fs" % (len(tokens), len(merges), meta.get("tokenizer.ggml.pre"), time.time() - started))

    full = build_tokenizer(tokens, merges, types, (3, 4))
    plain = build_tokenizer(tokens, merges, types, (4,))
    splitter = regex.compile(SPLIT)
    bytes_only = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)

    texts = FIXED + random_texts(4000, 20261107) + file_chunks(root)
    cases = []
    split_disagree = 0
    for text in texts:
        ids = full.encode(text, add_special_tokens=False).ids
        plain_ids = plain.encode(text, add_special_tokens=False).ids
        pieces = [p for p, _ in full.pre_tokenizer.pre_tokenize_str(text)]
        mapped = [bytes_only.pre_tokenize_str(p)[0][0] for p in splitter.findall(text)]
        if mapped != pieces:
            split_disagree += 1
        cases.append({"text": text, "ids": ids, "plain": plain_ids})
    print("reference: %d texts encoded, python regex vs tokenizers split disagreements: %d" % (len(cases), split_disagree))

    template = make_template(meta["tokenizer.chat_template"])
    chats = []
    for chat in CHATS:
        entry = {"messages": chat["messages"], "tools": chat.get("tools"), "options": chat.get("options", {})}
        try:
            prompt = render_chat(template, chat)
            entry["prompt"] = prompt
            entry["ids"] = full.encode(prompt, add_special_tokens=False).ids
        except Exception as problem:
            entry["error"] = str(problem)
        chats.append(entry)
    print("reference: %d chats rendered with jinja2, %d refused" % (len(chats), sum(1 for c in chats if "error" in c)))

    Path(out_path).write_text(json.dumps({"cases": cases, "chats": chats}, ensure_ascii=False), encoding="utf-8")
    print("reference written to %s in %.1fs" % (out_path, time.time() - started))


if __name__ == "__main__":
    main()
