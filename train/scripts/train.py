import json
import os
import random
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments


def say(kind, **values):
    print("@@" + json.dumps({"kind": kind, **values}), flush=True)


def loadModel(name, **options):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    try:
        return AutoModelForImageTextToText.from_pretrained(name, **options)
    except (ValueError, KeyError):
        return AutoModelForCausalLM.from_pretrained(name, **options)


def block(lang, code):
    return f"```{lang}\n{code}\n```"


# tres clases de ejemplo: seguir tus archivos linea a linea, tu codigo exacto para un pedido y las clases de los maestros
def examples(data, stylePairs):
    pieces = [json.loads(l) for l in open(os.path.join(data, "pieces.jsonl"), encoding="utf-8")]
    byId = {p["id"]: p for p in pieces}
    out = []
    for p in pieces:
        ask = f"Continue {p['file']} from here:\n{block(p['lang'], p['before'])}" if p["before"].strip() else f"Write the start of {p['file']}."
        out.append([{"role": "user", "content": ask}, {"role": "assistant", "content": block(p["lang"], p["code"])}])
    lessons = 0
    taught = os.path.join(data, "teacher.jsonl")
    if os.path.exists(taught):
        for line in open(taught, encoding="utf-8"):
            t = json.loads(line)
            p = byId.get(t["id"])
            if not p:
                continue
            if stylePairs and t.get("instruction"):
                out.append([{"role": "user", "content": t["instruction"]}, {"role": "assistant", "content": block(p["lang"], p["code"])}])
            for lesson in t.get("lessons", []):
                if isinstance(lesson, dict) and lesson.get("question") and lesson.get("answer"):
                    out.append([{"role": "user", "content": f"{lesson['question']}\n\n{block(p['lang'], p['code'])}"}, {"role": "assistant", "content": lesson["answer"]}])
                    lessons += 1
    sent = os.path.join(data, "external.jsonl")
    outside = 0
    if os.path.exists(sent):
        for line in open(sent, encoding="utf-8"):
            e = json.loads(line)
            ask = e.get("instruction") or e.get("question") or ""
            if e.get("code"):
                ask = f"{ask}\n\n{block(e.get('lang', ''), e['code'])}"
            if ask.strip() and e.get("answer"):
                out.append([{"role": "user", "content": ask}, {"role": "assistant", "content": e["answer"]}])
                outside += 1
    print(f"{len(pieces)} pieces of code, {lessons} lessons from the teacher, {outside} sent through the lessons API, {len(out)} examples", flush=True)
    return out


def chat(tokenizer, messages, prompt):
    options = {"tokenize": False, "add_generation_prompt": prompt}
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **options)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **options)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


class Progress(TrainerCallback):
    def __init__(self, tokens):
        self.tokens = tokens
        self.started = 0.0
        self.loss = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.loss = logs["loss"]
        if logs and "eval_loss" in logs:
            say("eval", loss=logs["eval_loss"])

    def on_step_end(self, args, state, control, **kwargs):
        spent = time.time() - self.started
        done = state.global_step
        total = state.max_steps
        rate = self.tokens * done / max(1, total) / max(1e-6, spent)
        say("step", done=done, total=total, seconds=spent, left=spent / max(1, done) * (total - done), loss=self.loss, tokensPerSecond=rate)


def main():
    config = json.load(open(sys.argv[sys.argv.index("--config") + 1], encoding="utf-8"))
    setup = config["train"]
    quick = config.get("quick", False)
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found for training.")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}, {free / 2**30:.1f} of {total / 2**30:.1f} GB free", flush=True)
    if free < 12 * 2**30:
        sys.exit("The GPU needs about 12 GB free: press Free the GPU and start again.")

    base = config["baseModel"]
    say("phase", text="Reading the examples")
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = examples(config["dataDir"], config["stylePairs"])
    random.seed(7)
    random.shuffle(rows)
    if quick:
        rows = rows[:80]
    limit = setup["maxLength"]
    encoded = []
    for messages in rows:
        full = chat(tokenizer, messages, False)
        head = chat(tokenizer, messages[:1], True)
        if len(head) >= limit:
            continue
        full = full[:limit]
        labels = [-100] * len(head) + full[len(head):]
        encoded.append({"input_ids": full, "attention_mask": [1] * len(full), "labels": labels[:len(full)]})
    if len(encoded) < 4:
        sys.exit("Not enough examples: collect some code first.")
    split = max(1, int(len(encoded) * setup.get("evalShare", 0.05)))
    trainSet = Dataset.from_list(encoded[split:])
    evalSet = Dataset.from_list(encoded[:split])
    epochs = 1 if quick else setup["epochs"]
    steps = 10 if quick else -1
    tokens = sum(len(e["input_ids"]) for e in encoded[split:]) * epochs
    if quick:
        tokens = sum(len(e["input_ids"]) for e in encoded[split:split + 10 * setup["accumulate"]])

    say("phase", text="Loading the base model")
    # hasta unos 10B entra en bf16 en 24 GB y entrena mejor; mas grande va en 4 bits
    fourBit = setup.get("fourBit", "auto")
    fourBit = fourBit is True or (fourBit == "auto" and any(s in base.lower() for s in ("14b", "27b", "32b", "35b")))
    options = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if fourBit:
        options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = loadModel(base, **options)
    if fourBit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    # solo las capas del lenguaje: el lector de fotos y la capa que adivina tokens quedan como estan
    names = [name for name, module in model.named_modules() if (isinstance(module, torch.nn.Linear) or type(module).__name__.startswith("Linear"))
             and "visual" not in name and "vision" not in name and "lm_head" not in name and "mtp" not in name]
    lora = LoraConfig(r=setup["rank"], lora_alpha=setup["alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=names)
    model = get_peft_model(model, lora)

    run = config["runDir"]
    args = TrainingArguments(
        output_dir=run,
        num_train_epochs=epochs,
        max_steps=steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=setup["accumulate"],
        learning_rate=setup["learningRate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        eval_strategy="no" if quick else "epoch",
        save_strategy="no",
        optim="paged_adamw_8bit" if fourBit else "adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    say("phase", text="Training")
    trainer = Trainer(model=model, args=args, train_dataset=trainSet, eval_dataset=evalSet, data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100), callbacks=[Progress(tokens)])
    trainer.train()
    adapter = os.path.join(run, "adapter")
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    say("done", adapter=adapter)


if __name__ == "__main__":
    main()
