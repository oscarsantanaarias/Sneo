import glob
import json
import os
import shutil
import subprocess
import sys

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import loadModel, say

config = json.load(open(sys.argv[sys.argv.index("--config") + 1], encoding="utf-8"))
base = config["baseModel"]
adapter = config["adapter"]
merged = os.path.join(os.path.dirname(adapter), "merged")
converter = config["converter"]
if not os.path.exists(os.path.join(converter, "convert_hf_to_gguf.py")):
    sys.exit("Pick the folder that has convert_hf_to_gguf.py in the export settings.")

# junta el adaptador con el modelo base en la cpu asi no hace falta vram libre
say("phase", text="Joining what it learned with the base model")
model = loadModel(base, dtype=torch.bfloat16, device_map={"": "cpu"})
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(merged, safe_serialization=True, max_shard_size="4GB")
del model

# el convertidor lee el tokenizador, la plantilla y los archivos del modelo base tal cual
folder = snapshot_download(base, allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"])
for path in glob.glob(os.path.join(folder, "*")):
    name = os.path.basename(path)
    if name.endswith(".index.json") or name == "config.json":
        continue
    shutil.copy(path, os.path.join(merged, name))

os.makedirs(os.path.dirname(config["target"]), exist_ok=True)
say("phase", text="Writing the GGUF file")
env = dict(os.environ, PYTHONPATH=os.path.join(converter, "gguf-py"))
code = subprocess.call([sys.executable, os.path.join(converter, "convert_hf_to_gguf.py"), merged, "--outtype", config["type"], "--outfile", config["target"]], env=env)
shutil.rmtree(merged, ignore_errors=True)
if code != 0:
    sys.exit("The GGUF conversion failed, the lines above say why.")
say("done", target=config["target"])
