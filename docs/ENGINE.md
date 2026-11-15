# Sneo engine

Sneo runs Qwen3.8-Flash-Next (`qwen4exp`, 48 layers, 512 experts per layer, 10 routed per token; the coder variant has 256) on one NVIDIA or AMD GPU, using the CPU, RAM and the SSD for whatever does not fit in video memory. Several PCs can be joined into a pool.

## Hardware

- CPU: any x86-64 processor with AVX2, FMA and F16C (Intel since Haswell, AMD since Ryzen). The engine checks this at startup.
- Windows 10 can move background processes to the efficiency cores (power throttling) when nobody is using the PC; on hybrid CPUs that made decoding 15 times slower. The engine opts out of it at startup.
- GPU: NVIDIA. RTX 30 series and newer (Ampere tensor cores) run every path. RTX 20 and GTX 10 series run the same decode kernels and read prompts with int8 dot product (dp4a) tile kernels instead of tensor cores (see Prefill). The kernels are compiled for the local GPU (cubin, PTX as fallback), so older drivers load them. The dense part alone takes 3.6 GB, so 8 GB of VRAM is the practical minimum. Without a usable GPU the engine runs on the CPU.
- GPU: AMD, used when there is no NVIDIA GPU. The engine loads amdhip64 in place of nvcuda: the HIP module calls take the same arguments as the CUDA driver calls, so the same tables hold either one. The same kernel source is compiled with hiprtc for the local chip, after a short prelude that maps the warp calls (shuffles, ballot, warp sync) to their HIP forms, dp4a to the chip's dot instruction (sdot4 on Vega 20, the RX 5500 and 5300, RDNA 2 and CDNA, sudot4 on RDNA 3 and 4, plain math on Polaris, Vega 10 and the RX 5700) and the few lines of inline PTX to plain C++. The kernels are written for warps of 32 threads. RDNA cards run waves of 32, so a warp is a wave. Polaris, Vega and CDNA run waves of 64: there each wave holds two warps, the shuffles stay inside their half (width 32), the ballot reads the half of the 64 bit mask that belongs to the thread's warp and the warp sync is a wave barrier with a fence, so the same kernels give the same results on either wave size. AMD cards run the paths of cards without tensor cores (dp4a tiles for prompts, no mma). Host memory the GPU polls is allocated coherent, so the GPU sees the CPU experts finish while a kernel waits. There is no second GPU helper on AMD. Not run on a real AMD card yet.

## Models

Any Qwen3.8-Flash-Next GGUF loads: IQ1_M, IQ2, IQ3, IQ4, Q4_K, Q5_K, Q6_K, Q8_0 and Q5_1 tensors run on the GPU, and the CPU expert path takes IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0 and Q5_1 too. The expert count comes from the file. Files whose name starts with `mmproj` (vision) are skipped. Settings, Engine lists the models in a folder, shows the one running and loads another; the engine remembers the last folder and model in `pool.json` and opens them at start. Switching models recreates the CUDA context: on a GTX 1070 a second load in the same context ran out of memory while pinning host RAM and left the context broken.

## Where the model lives

| Part | Where | Notes |
| --- | --- | --- |
| Dense layers (attention, gated delta net, hyper connections, routers, shared experts, output head) | VRAM | about 3.6 GB |
| KV cache | VRAM | sized by the context set in the pool |
| Hot experts | VRAM | as many as fit, about 200 of 512 per layer on a 24 GB card |
| Next experts | pinned RAM | read by the GPU over PCIe; Windows lets the GPU read about half of the RAM this way |
| Experts for the CPU | RAM, locked in the working set | the CPU computes the experts it is given straight from RAM |
| Everything else | SSD | the expert pack (`.mxpack`) and the n-gram table, read on demand |

The order is always VRAM, then RAM, then SSD. The memory limits per machine come from the Pool view (`pool.json` slots).

## Decode

Each token runs the 48 layers as one CUDA graph. For every layer the GPU computes the dense part and the experts already in VRAM, and the routed experts that missed VRAM are split between the GPU, which reads them from pinned RAM over PCIe, and the CPU, which computes them at the same time. A single token sends half of its misses to the GPU (`warmShare`, in eighths, default 4). A verify pass counts the misses of all its tokens in the layer: the CPU takes what it can finish while the GPU works on the experts in VRAM and the GPU reads only the excess, (misses - 1) * 3 * `warmShare` / 32 routes, and every route of an expert goes to the same side so its weights are read or decoded once. A miss costs about the same on either side (about 45 us on 16 CPU threads, about 60 us of PCIe for the GPU), so what matters is that neither waits for the other. The GPU and the CPU meet through flags in mapped memory, so there is no full synchronization per layer.

The quantized matrix kernels for one to four rows are compiled per quantization type and row count and read the weights straight from global memory with aligned vector loads (on GTX 10 and RTX 20 each row is first copied to shared memory with coalesced loads instead: on a GTX 1070 the coder's Q6_K projections went from 92 to 168 GB/s). For three and four rows each warp works on two weight rows at once, so the activations are read half as often; the arithmetic is the same, so the results are bitwise identical to the one row kernels. Activations are quantized to int8 in two levels (a value and its residual, about 15 bits in all). A single level (`residual = false`) makes the dense part 12 % faster for one row and 29 % for four rows, but on long code it measured +0.8 % loss, so two levels stay the default. The CPU expert kernels are also specialized per type. For the IQ3_XXS, IQ3_S and IQ2_S grids they decode the signs, scales and grid indexes of a whole block of 256 weights at once, put the grid values straight into the vector register and read both activation levels from one interleaved buffer, with the same results bit for bit as decoding group by group; one route (gate, up and down of one expert for one token) takes about 330 us on one P core of an i9-12900K and about 45 us on its 16 cores. The CPU thread pool waits for the chunks of a job, not for every worker, so a worker that Windows took off its core does not hold a layer.

The expert cache in VRAM starts from a heat profile and keeps adapting to the conversation: every 32 tokens the experts used most by the generation replace the coldest ones. Uploads run on a separate copy stream, so decoding never waits for them.

## Routing modes

`fastRouting` in `pool.json` (on by default):

- **fast**: routes ranked 8 to 10 whose expert is not in VRAM are skipped and the remaining weights are renormalized. It applies only while fewer than 90 % of the experts are in VRAM. About 7 % faster decode on long prompts. Output stays very close to the model; on medium and long prompts the greedy text matched the exact one, but with a cold cache (few experts in VRAM) the text can change.
- **exact**: every routed expert is computed. Output matches the reference forward.

## Speculation

`speculate` in `pool.json` (on by default). The engine loads the model's own draft layer (`modelos/Qwen3.8-Flash-Next.draft-q2.gguf`, built with `tools/fetchdraft` and `tools/draftbuild`, about 0.9 GB of VRAM). Each draft cell pairs the main model's final residual at one position with the next token, and the layer's own residual feeds the next draft. The layer absorbs every main row whose next token is known (the prompt in blocks of 256 rows, then every decoded row), so its KV cache always covers the whole context. The rows of a verify pass are absorbed inside the verify graph itself, with no extra round trip to the host. The draft experts stay in 2 bits: 4-bit experts were measured at only +0.3 to +0.6 points of agreement for 0.7 GB more VRAM.

Per step the draft proposes up to three tokens and stops at the first one whose probability is under 0.6; when even the first one is unsure, the step decodes a single token. The picked token and its probability are computed on the GPU. The draft only scores the 65536 most frequent tokens of English text and code (`profiles/<model>.draftvocab`, built with `tools/draftvocab`), which cover 99 to 99.5 % of English and code and about 96 % of Spanish; its output head is a quarter of the full one, proposals got 37 % cheaper and output speed about 8 % higher with the same acceptance. Without the file the draft scores the whole vocabulary. The verify path checks up to four tokens in one graph and its logits are bitwise identical to decoding them one by one. When a draft is rejected, the recurrent state (gated delta net state and convolution history, per layer embedding history) is rebuilt up to the last accepted token from the inputs saved during the verify pass, about 1 ms instead of a second forward. In exact routing the text is the same with and without speculation.

On the benchmark suites it gives about +25 to 30 % output speed on long and code prompts; drafts that pass the 0.6 gate are accepted 70 to 90 % of the time.

## KV cache

Only 12 of the 48 layers use attention, the other 36 keep a fixed size recurrent state. Keys, values and indexer keys are stored in int8 with one scale per 32 values, 1296 bytes per token and attention layer, about 1 GB at a 64K context; the memory saved over FP16 goes to resident experts. Perplexity is unchanged (`gpugen --perplexity`).

## Prefill

Long prompts go through the layers in blocks of up to 8192 tokens (2048 on GTX 10 and RTX 20); the dense part of each block runs 1024 tokens at a time. Prompts of 4096 tokens or more borrow video memory from the recent part of the expert cache: those pages are copied to pinned RAM, mapped together as one prompt area, and copied back unchanged afterwards, so a prompt of up to about 24K tokens on a 24 GB card is read in a single block and every streamed expert crosses PCIe once. Every expert runs on the GPU: resident ones from VRAM, the others streamed from pinned RAM or the expert pack through a double buffered pinned ring. With the NVIDIA matrix library (cuBLAS, RTX 30 and newer) the BF16 matrices run in it with the activations split in two BF16 terms (the same precision as before) and the quantized dense matrices of each layer are unpacked once per block to FP16 and multiplied in FP16 with FP32 sums; without it they use the engine's BF16 and int8 tensor core kernels. Prompt attention reads each cached key and value once for all the query heads that share it. Prompt processing is deterministic: the expert outputs of a block are added as 64-bit fixed point integers (exact in any order) and split matrix products add their parts in a fixed order, so the same prompt always leaves the same state after it is read. Writing is not bit exact across runs when experts are missed: the CPU and the GPU round slightly differently and the expert cache moves experts between them over time, so a near tie between two tokens can go either way.

Without tensor cores the dense matrices go through `dotRows` (one instance per weight format): 64 rows by 64 tokens per block of 128 threads, 4 rows by 8 tokens per thread, weights unpacked to int8 column major in shared memory so each thread reads its rows with 128 bit loads, integer sums turned into floats with an exact add instead of the quarter rate conversion. Experts and dense blocks of 128 tokens or more use `dotRowsW` (256 threads, 128 tokens per block, so an expert with fewer than 128 tokens unpacks its weights once). BF16 matrices use `plainTile`. The dense part of each block runs 512 tokens at a time. Batched prompts on these cards use one int8 level for the activations: the logits move about as much as between the existing paths and the greedy text stays the same.

Three Pascal details matter. Plain global loads go to L2 only, so the IQ grid tables are read with `__ldg` (an IQ3_XXS group does about twelve dependent lookups; this halved the expert kernels). Windows keeps launches in the driver until something flushes them, so the prompt path queries the stream after each expert batch, after the resident experts and after each dense block. And the cold experts of the next two batches are requested from Windows ahead of time (`PrefetchVirtualMemory`), so experts that are not in RAM arrive in large reads instead of page faults.

On a GTX 1070 (PCIe 3.0, 11.9 GB/s, 32 GB of RAM) a 1808 token prompt reads in 7.6 s with the 256 expert coder file and 16 s with IQ3_XXS, whose cold experts do not fit in RAM; streaming the experts over PCIe alone takes about 1.8 s with the coder. Tried and dropped: copying pinned experts to the GPU without the ring (the first prompt pays about 1.5 s while Windows makes the pinned RAM resident), L1 caching of every global load (`-dlcm=ca`, prompts 3 % faster but writing slower), 4096 token blocks (long prompts 8 % faster, shorter ones slower for two fewer resident experts) and two attention heads per block (same time: the redundant key reads already hit L2; the cost is the per cell warp reductions and int8 to float conversions).

## Prompt reuse

A new request continues from the previous one when the tokens held are a prefix of it. Otherwise the engine restores a snapshot of the recurrent state (gated delta net state, convolution history, per layer embedding history; the KV cache stays in place): one at the end of the system prompt (when it has 256 tokens or more) and one at the end of the last user message. So a resent or regenerated message, the next turn of a chat and a history whose earlier answers come back re-rendered differently only read the new tokens: on a GTX 1070 a 1845 token prompt took 9.9 s the first time, 0.5 s resent and 1.2 s for the next turn. Snapshots live in VRAM and fall back to pinned RAM when VRAM is full; a failed snapshot is logged.

On a GTX 1070 the coder writes 16.7 tok/s in `decodeprof` and 17.3 in the app (IQ3_XXS 17.8). Per token the GPU part takes about 29 ms and the CPU experts about 26 ms, one after the other in every layer; `warmShare` 2 is still the best split there.

## Pool compute

A PC in the pool with an NVIDIA GPU can take the first layers of the model. The master gives it as many layers as fit in its GPU slot (at least two, so it also applies the per layer embedding at layer 1), ships those tensors once (cached on the node by name and size) and runs the rest of the layers, the head and the draft itself. Per step the master sends the positions, the token embeddings and the per layer embedding rows (20 KB per token) and gets back the residual streams after the node's last layer (40 KB per token, float32, so the result is the same as running the whole model here). Saves, restores, commits and resets of the recurrent state are mirrored on the node over the same connection, without waiting for an answer. The node loads its layers from the cache into its own GPU and answers on a second connection that starts with a ready message. If the node cannot take its layers, or drops while generating, the master loads the whole model again. Each step pays one network round trip, so on Wi-Fi the pool is slower than one PC that holds the model; it is meant for models or GPUs that do not fit on one machine.

## Heat profile

`profiles/<model>.heat` decides which experts start in VRAM. It is built from decode routing (`tools/heatgen`): the model answers about 160 varied prompts and the experts it uses while writing are counted. Prompt routing is not used, because the experts that read a prompt are not the ones that write the answer.

## Pool settings (`pool.json`)

| Key | Meaning |
| --- | --- |
| `backend` | `auto`, `gpu` or `cpu` |
| `context` | context length; the KV cache in VRAM grows with it |
| `fastRouting` | skip missed tail experts (see above) |
| `speculate` | draft and verify several tokens per step (on by default) |
| `draftLength` | most drafts per step, 1 to 4 (default 3; the verify pass checks one token more) |
| `draftSure` | probability a draft needs to be proposed (default 0.6) |
| `slots` | ordered memory limits per machine and tier (`gpu`, `ram`, `ssd`) in GB, percent or "keep free" |
| `model`, `folder` | last model loaded and the folder scanned for models (set from Settings) |

## Tools

| Tool | Use |
| --- | --- |
| `tools/bench.exe` | API benchmark (short, code, long suites); run it from the project folder |
| `tools/decodeprof.exe` | decode profile: time per token in blocks, GPU wait, CPU experts, VRAM hit rate, cost of verifying 2 to 8 tokens; with a model path, a VRAM cap and `profile` it also times every prompt kernel |
| `gpugen.exe` | checks the GPU forward against the CPU reference (`--verify`, `--drafts`, `--tail`, `--super`); `--commit` checks bitwise that verify rows and the state rebuilt after a rejected draft match plain decoding; `--dump` writes deterministic logits for before/after comparisons; `--plainbench` times the dense matrix kernels |
| `tools/heatgen.exe` | builds the decode heat profile |
| `tools/fetchdraft.exe`, `tools/draftbuild.exe` | download and convert the draft layer for speculative decoding |
| `tools/draftvocab.exe` | ranks tokens by frequency in a set of folders and writes the draft vocabulary, with coverage on test folders |
| `gpugen.exe --perplexity` | perplexity of fixed texts (`--text <file> --span N` cuts a frozen file in chunks, `--losses` writes the loss of every token); long code is very sensitive to any numeric change, so compare several texts and per token |
