# Sneo

Sneo runs Qwen3.8-Flash-Next, Qwen3.6, Qwen3.5 and Gemma 4 models on one NVIDIA or AMD GPU and uses the CPU, RAM and SSD for whatever does not fit in video memory, so a 70 GB model runs on a 24 GB card. It reads the model's GGUF files directly: there is no conversion step and no second copy of the model.

Made by Oscar Santana, a developer from the Dominican Republic: [LinkedIn](https://www.linkedin.com/in/oscar-santana-b1503436b/) · [sneo.dev](https://sneo.dev)

<a href="https://buymeacoffee.com/oscaratr20d" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" width="180" />
</a>

## What the app does

- **Coding agent.** It reads, searches, edits and runs your projects, can hand parts of a task to subagents and understands the tool names other agents use. Every step shows up in a timeline, and every edit as a side by side change you can open in a large window.
- **Reverse engineering with IDA and Ghidra.** The agent talks to IDA and Ghidra directly through a small TCP bridge inside each one (IDA on port 5073, Ghidra on 5093): it decompiles functions, follows cross references, callers and vtables, searches strings and bytes, renames and retypes what it works out, maps whole subsystems and searches the decompiled code by meaning.
- **Live status.** The chat says what the model is doing at every moment: reading the prompt (new and reused tokens, speed), writing a tool call (which tool, which file, how far it got) and the writing speed while it answers.
- **Memory in the open.** The model menu shows the VRAM, RAM and disk the loaded model uses right now, exact to the byte. Messages wait until a model is loaded.
- **Agents on several models.** The Agents view keeps one conversation per agent (orchestrator, coder, researcher, reviewer or your own), each one on its own model: Sneo on this PC, a second local model, Claude, OpenAI, DeepSeek, Gemini, OpenRouter or any chat API like OpenAI's, and Claude Code installed on the PC with your own subscription, no API key. A command center sends an instruction to every agent or to one, any answer can be passed on to another agent, and a team run lets a planner split a task, the agents work it in parallel on their models and a lead merge the answers, drawn live as a graph or as an office where each agent sits at its desk and walks to another one when it hands over work. Keys stay on the PC.
- **Train on your code.** The Train view fine-tunes a model on your own projects: your code is cut into pieces, another model (one of your agents) writes the lessons, and the training runs on your card with an estimate of how long it takes.
- **Two models at once.** Pick both models, the card of the second and, on a shared card, the video memory of each (at least 8 GB each), then load them together; to change them, unload and load again. Windows lets the cards fix at most half of the RAM and both engines share that, so Sneo splits the fixed RAM and the RAM between them before loading and the rest is read from the SSD. Measured with two IQ3 models: IQ3_S on an RTX 3090 writes 76 to 87 tok/s with IQ3_XXS loaded on a GTX 1070 next to it (19 to 21 tok/s there). Engines on one card take turns writing, on two cards they write at the same time. Unloading gives the first model its whole card and settings back.
- **Two PCs as one.** A second PC in the pool, on a cable or Wi-Fi, runs the first layers on its own card. Little crosses the network: the rows go as the model's own quantized blocks (exact, a quarter of the size), the answer comes back in half precision when it fits, checking drafts does not copy any state, and a long prompt goes in parts so the network and both cards work at the same time. Measured on one PC with the network imitated, over a gigabit cable the first token of a 24K token prompt came after 11.8 s instead of 25.7 s and writing went from 68 to 72-76 tok/s; over 300 Mbit Wi-Fi the same prompt took 19.5 s instead of 53 s.
- **From your phone.** Turn on Phone, scan the code and the phone gets a page you can add to the home screen: chat with the agent and approve what it asks, the agent conversations, and the engine with its models. Each phone is paired with its own key that you can remove, and only the home network (or a VPN like Tailscale) can reach it.
- **English or Spanish.** One setting switches the whole app, agent answers included.

## Models

### What Sneo runs

| Model | Kind | Size | Where to get it |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next IQ3_XXS | 512 experts | 71 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| Qwen3.8-Flash-Next IQ3_S | 512 experts | 78 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| Qwen3.8-Flash-Next UD-Q4_K_XL | 512 experts | 104 GB | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| Qwen3.8-Flash-Next Coder IQ1_M | 256 experts, made for code | 54 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) |
| Swift 1.5 Qwen3.8-Flash-Next IQ2_XS | Fine-tune of Qwen3.8-Flash-Next, 512 experts | 63 GB | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), its IQ3_XXS and IQ3_S files load too |
| Qwen3.6-35B-A3B UD-Q4_K_XL | Mixture of experts, 3B active | 21 GB | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), or `qwen3.6:35b` in Ollama |
| Qwen3.8 27B | Dense | 16 GB | `qwen3.8:27b` in Ollama |
| Qwen3.6 27B | Dense | 16 GB | `qwen3.6:27b` in Ollama |
| Qwen3.5 4B | Dense | 2.5 GB | `qwen3.5:4b` in Ollama |
| Gemma 4 26B | Mixture of experts | 16 GB | `gemma4:26b` in Ollama, its draft assistant comes along and makes writing faster |
| Gemma 4 31B | Dense | 17 GB | `gemma4:31b` in Ollama, its draft assistant comes along and makes writing faster |
| Hy4-flash Q4_K_M (preview) | Tencent, 512 experts, 12B active | 157 GB | [AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF), see [Hy4-flash](#hy4-flash-preview) |

The same models as GGUF files from Hugging Face load too, and so do other quantizations of them, as long as their tensors are of these types: IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0, Q4_K, Q5_K, Q5_1, Q6_K, Q8_0, NVFP4, MXFP4, F16, BF16 and F32 (for example the usual Q4_K_M, Q5_K_M, Q6_K and Q8_0 files, and the NVFP4 ones that keep a scale per expert).

All of them are free. **Get models** in the model menu downloads Qwen3.8-Flash-Next and Qwen3.6-35B-A3B from Hugging Face and marks the one that fits the PC best. A model of another architecture still shows up in the list, and loading it says that it is not supported yet.

For Qwen3.8-Flash-Next, bigger files answer better and need more memory. As a guide, on a PC with an 8 GB GPU and 32 GB of RAM:

| Model | Size | How it runs |
| --- | --- | --- |
| Coder IQ1_M | 54 GB | Fits in VRAM and RAM, the fastest |
| IQ3_XXS | 71 GB | Better answers; part of the experts are read from the SSD |
| IQ3_S | 78 GB | A little better again, more SSD reads |
| UD-Q4_K_XL | 104 GB | Best of these, wants 64 GB of RAM or more |

A model split in parts (`-00001-of-00002.gguf`, `-00002-of-00002.gguf`, ...) needs all its parts.

### Hy4-flash (preview)

Tencent's Hy4-flash runs in Sneo. It is big: 157 GB at Q4_K_M and 12B active parameters per token, so it wants a 24 GB card and 96 GB of RAM.

- Download both parts of the Q4_K_M from [AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) into one folder.
- For faster writing, make a `dfly2` folder next to them and put `model-00148-of-00148.safetensors` from [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash) in it (1.7 GB). That file holds DFly2, the model's own draft layers. Without it Hy4 still runs, only slower.
- The first load after an update builds the GPU kernels, about a minute more; after that a load takes about a minute. The experts used most stay on the card and the list is kept in `profiles\Hy4-flash-Q4_K_M.hy4heat`, so it starts warm and gets faster as you use it.

Each step of writing goes to the card in one launch: the card stops at every layer that needs experts from RAM, the CPU computes them and hands them back through shared memory, without going back to the driver. The experts you use move to the card in the background while it writes, so the card keeps the ones your work needs. On an RTX 3090 with 96 GB of DDR5 it reads long prompts at 400 to 500 tok/s, writes code at 60 to 75 tok/s, code at about 60 tok/s with a 10K token prompt like the coding agent's, and prose at about 40 tok/s after a 10K token prompt. Tool calls work with the coding agent.

### Where to put them

Anywhere. The only rule is that the parts of one model stay together in the same folder.

- The simplest is the `modelos` folder next to `master.exe`: Sneo finds the files there without asking.
- If the models are somewhere else, open **Settings > Engine** and press **Browse** (or **Change folder** in the model menu under the message box). Sneo looks inside that folder and its subfolders, so one subfolder per model works too.
- Files whose name starts with `mmproj` (vision add-ons) are ignored.

Example:

```
D:\Models\
    IQ3_XXS\
        Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
        Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
    coder-IQ1_M\
        Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
        Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
```

### Models you already downloaded with Ollama or Strata

No need to download them again: Sneo runs them straight from the other app's files, so nothing is copied.

- **Ollama:** nothing to choose. Sneo lists the models Ollama has downloaded (in `%USERPROFILE%\.ollama\models`, or in the folder set in `OLLAMA_MODELS` if you moved them) under the names they have there, for example `qwen3.8:27b`, marked *From another app*.
- **Strata, or any other folder with GGUF files:** choose that folder as Sneo's models folder, in **Settings > Engine > Models folder** and **Scan**, or with **Change folder...** in the model menu.

### Files Sneo prepares by itself

Two extra files make Sneo faster. Nobody has to make them by hand: after a model loads, Sneo builds the ones that help in the background, at low disk priority, and shows the progress in the model menu. Both go to the `modelos` folder, never to your models folder, and both are used from the next time the model loads.

| File | What it does | When Sneo makes it |
| --- | --- | --- |
| `Qwen3.8-Flash-Next.draft-q2.gguf` (0.9 GB) | The draft layer for speculative decoding: it guesses the next tokens and the model checks several at once, about 25 to 30 % faster writing on code | For the 512 expert models, when it is missing. It downloads the layer from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (about 5 GB), converts it and deletes the download |
| `<model>.mxpack` (as big as the model's experts) | The experts rewritten so each one is a single aligned block, read from the SSD in one go instead of page by page | When the model's experts do not fit in your RAM (for example IQ3_XXS with 32 GB). A cold 1808 token prompt went from 24 to 19 s on a GTX 1070 |

The release zip already brings the draft layer in `modelos`, so Sneo does not download it. That file is part of Qwen3.8-Flash-Next, so it keeps the model's license, the Qwen Community License 1.0, in `modelos\QWEN_LICENSE.txt`.

To make them by hand: `tools\fetchdraft.exe` then `tools\draftbuild.exe` for the draft layer, `tools\repack.exe <model.gguf>` for the pack.

### Memory

Opening Sneo does not load anything into RAM or VRAM. The model loads with the first message (10 to 15 s once) and stays loaded. **Free memory** in the model menu releases it, and the next message loads it again.

## Running

- **Download:** the [Releases](https://github.com/oscarsantanaarias/Sneo/releases) page has a zip with the engine, its tools and the desktop app already built. Unzip it and run `start.bat`: Visual Studio and Rust are not needed. The first start still gets Node.js if the PC has none, the UI packages, NVRTC and, on NVIDIA cards, cuBLAS.
- **Sneo from the source:** `start.bat`. The first time it gets what is missing by itself: a private copy of Node.js if the PC has none, the UI packages, the page build, the NVIDIA kernel compiler (NVRTC) for the engine, on NVIDIA cards the NVIDIA matrix library (cuBLAS, about 530 MB, used to read long prompts faster; without it the engine reads them with its own kernels) and, when the desktop app has not been built yet, Rust to build it once. If the desktop app cannot be built on that PC, Sneo opens in the browser instead. Then it starts the engine and the app.
- **Engine only:** `master.exe 8000 - - modelos` (the APIs below, no window).
- **Pool worker on another PC:** open Sneo there and join the pool from the Pool view; it starts the worker and opens the firewall by itself.
- **Build the engine:** `build.bat` (Visual Studio 2026, C++17). The GPU kernels are compiled at first start with NVRTC.

### Using it from other apps

The engine listens on `http://127.0.0.1:8000` (this PC only) and speaks two APIs:

| API | Endpoint | Point your client at |
|---|---|---|
| OpenAI chat completions | `POST /v1/chat/completions`, `GET /v1/models` | base URL `http://127.0.0.1:8000/v1` |
| Anthropic messages | `POST /v1/messages`, `POST /v1/messages/count_tokens` | base URL `http://127.0.0.1:8000` (for example `ANTHROPIC_BASE_URL=http://127.0.0.1:8000`) |

Both stream, take tools and return tool calls. Any API key works. On the messages API, thinking is off unless the request turns it on (`"thinking": {"type": "enabled", "budget_tokens": N}`), and the answer comes back as `thinking`, `text` and `tool_use` blocks.

## Hardware

- An x86-64 CPU with AVX2, FMA and F16C (Intel since Haswell, AMD since Ryzen).
- An NVIDIA GPU with 8 GB of VRAM or more, GTX 10 series or newer. RTX 30 series and newer use tensor cores, and RTX 50 cards read the prompts of NVFP4 models on their FP4 tensor cores (not run on a real RTX 50 yet; SNEO_FP4_TENSOR=0 turns it off); RTX 20 and GTX 10 series work too (tested on a GTX 1070). Without a usable GPU the engine runs on the CPU.
- Several NVIDIA cards in one PC work together: the main card runs every layer and the other one keeps the experts that do not fit and computes them, or runs the second model. Any mix from the GTX 10 series up works, as long as one driver runs all the cards: NVIDIA's 580 driver branch is the last one with GTX 10 support, so a PC that mixes a GTX 10 card with a newer one needs a 580 driver (tested with 581.80 on an RTX 3090 with a GTX 1070).
- Or, on a PC without an NVIDIA GPU, an AMD Radeon with 8 GB of VRAM or more: RX 400 and 500 (Polaris, for example the RX 480 and RX 580 8 GB), RX Vega 56 and 64, Radeon VII, RX 5000 (RDNA 1), RX 6000 (RDNA 2), RX 7000 (RDNA 3) and RX 9000 (RDNA 4), and the Radeon Pro cards of those families. It needs the AMD driver and the AMD HIP SDK for Windows, which brings the kernel compiler (hiprtc); the engine finds it through HIP_PATH, or copy the SDK's hiprtc and amd_comgr DLLs into runtime. AMD cards take the same paths as NVIDIA cards without tensor cores; Polaris, Vega and the RX 5700 have no int8 dot instruction, so they do that math in plain instructions and are slower per byte. AMD's HIP runtime for Windows officially lists only recent cards, so an older card depends on the driver it has. This has not been run on a real AMD card yet. If the kernels come out for the wrong chip, set SNEO_GPU_ARCH to the chip name, for example gfx803 for an RX 580, gfx1010 for an RX 5700 XT or gfx1100 for an RX 7900 XTX.
- One PC runs one kind of card, NVIDIA or AMD. In a pool each PC uses its own cards, so an NVIDIA PC and an AMD PC can share one model.
- 32 GB of RAM or more and a fast SSD.

How the engine works inside: [docs/ENGINE.md](docs/ENGINE.md).

## License

MIT, see [LICENSE](LICENSE).
