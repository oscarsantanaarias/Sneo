import sys, json, jinja2
from tokenizers import Tokenizer
sys.stdout.reconfigure(encoding='utf-8')
source = open(sys.argv[1], encoding='utf-8').read()
source = source.replace('{%- generation -%}', '').replace('{%- endgeneration -%}', '')
env = jinja2.Environment(extensions=['jinja2.ext.loopcontrols'])
def tojson(x, indent=None, separators=None, sort_keys=False, ensure_ascii=False):
    return json.dumps(x, indent=indent, separators=separators, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
def fail(message):
    raise Exception(message)
env.filters['tojson'] = tojson
env.globals['raise_exception'] = fail
case = json.load(open(sys.argv[2], encoding='utf-8'))
text = env.from_string(source).render(messages=case['messages'], tools=case.get('tools'), reasoning_effort=case.get('effort'))
tok = Tokenizer.from_file(sys.argv[3])
ids = tok.encode(text, add_special_tokens=False).ids
print(len(ids))
print(text)
