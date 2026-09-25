# Jalapeño — Qwen3.8-Flash-Next on a DGX Spark, fast

![platform](https://img.shields.io/badge/target-DGX_Spark_GB10_%2F_sm121-76b900) ![weights](https://img.shields.io/badge/weights-Qwen_Community_License_1.0-blue) ![code](https://img.shields.io/badge/patches_%2B_tooling-Apache--2.0-orange)

A tuned vLLM stack that runs the 335 GB **Qwen3.8-Flash-Next** on a single
**NVIDIA DGX Spark** (GB10, 121 GB unified memory) as a full OpenAI-compatible
endpoint — tool calling, reasoning parsing, prefix caching, 200K context,
speculative decoding — together with the patch campaign ("Project Jalapeño")
that took single-stream decode from **24.2 → ~38 accepted tok/s (+57 %)**
with quality gates green throughout (13/13 battery, KL ≤ 0.012 vs BF16).

Everything here ran in production on a real Spark for weeks before being
published. No part of it is theoretical.

| | |
|---|---|
| Single-stream decode (accepted tok/s) | **37.3–38.0** (MTP=2, all optimizations) |
| … before the campaign, same stack | 24.2 |
| Concurrent streams (1 / 4 / 8 / 16) | 38.7 / 21.2 / 14.6 / 10.5 tok/s |
| Mixed agent fanout (up to 8 streams) | 23.8 tok/s aggregate · TTFT p50 1.4 s / p90 6.1 s · per-stream p50 8.6 tok/s · 0 errors / 76 requests |
| Inter-token latency p50 | 56.8 ms |
| TTFT p50 (short prompts) | ~300 ms |
| Context | 131,072 default · 204,800 in production |
| Boot | ~6.5 min (198 s weight load) |
| Decode power | ~32 W |

The full evidence chain — every promotion, what it won, what was tried and
rejected, and how the numbers were measured — is in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## What you're running

Qwen3.8-Flash-Next is a hybrid linear-attention MoE model: 48 MoE layers ×
512 routed experts, GDN + QSA attention (most layers are context-immune,
which is why 200K-token windows are practical), a built-in MTP draft head
(speculative decoding is nearly free), and an unusual 51.2B-parameter PLE
n-gram table. That table is why no published vLLM quant of this model fits
a single GPU — they all keep it at FP8 or wider — and quantizing it is where
this repo starts.

Served through vLLM with the reasoning parser, tool-call parser
(`qwen3_coder`), and auto tool choice enabled, so OpenAI-API clients and
agents work unmodified.

## Quickstart

You need: a DGX Spark (any 121 GB GB10), Docker with the NVIDIA runtime
(the Spark ships with it), ~110 GB of NVMe for weights, and
`pip install 'huggingface_hub[cli]'` for the download.

```bash
# 1. Weights (109 GB, public): the only single-GPU vLLM quant of this model.
hf download starkweatherdigital/qwen3.8-flash-next-nvfp4 \
   --local-dir /mnt/models/qwen3.8-flash-next-nvfp4

# 2. Engine image: the 11 patches applied over a digest-pinned vLLM base.
#    (The base is a 30 GB pull; the patch step itself is seconds.)
docker build -t jalapeno/vllm-gb10-flashnext:0.28-sm121-r9 .

# 3. Serve. Defaults are the production-validated configuration.
cp .env.example .env        # then set MODEL_DIR to the path from step 1
docker compose up -d
docker compose logs -f      # ~6.5 min to healthy — it's loading 76 GB of weights
```

First request:

```bash
curl -s http://localhost:8006/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next",
       "messages":[{"role":"user","content":"Name the capital of Australia. One word."}]}' \
| python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
# → Canberra
```

Then run the full acceptance gates (stdlib only, runs on the host):
`python3 serve/gates.py --url http://localhost:8006/v1` — coherence, a real
tool call, speed with GPU clocks logged, speculative acceptance, memory.

**Endpoints** (host network, port 8006): `/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/metrics` (Prometheus, including
speculative-decode acceptance and KV usage), `/health`, and — with
`SLEEP_MODE=1` as shipped — `/sleep`, `/wake_up`, `/is_sleeping`.

## Using it with your existing tools

Point anything OpenAI-compatible at `http://<spark>:8006/v1`, model name
`qwen3.8-flash-next`:

- **Chat UIs** (Open WebUI, LibreChat, …) — add an OpenAI connection and it
  just works, reasoning tags and tool calls already parsed server-side.
- **Agents and coding tools** — anything speaking the OpenAI chat API works
  unmodified. Prefix caching makes multi-turn agents cheap: 56–70 % cache
  hit rate on conversation-shaped traffic, a cached turn drops from ~10 s to
  ~2–3 s of prefill. Under parallel agent load expect the fanout row in the
  table above — aggregate throughput holds up and no stream starves.
- **Sharing the box** — `POST /sleep` releases the GPU for another job
  (a vision sidecar, a training run); `POST /wake_up` brings the engine back
  without restarting the container.

### Security note

The compose file uses host networking and the server binds `0.0.0.0` with
**no API key** — convenient on a trusted home LAN, exposed anywhere else
(anything that can reach the Spark can use the model *and* read
`/metrics`, and `/sleep` lets it evict your engine). If the Spark sits
outside your trust boundary, set `VLLM_API_KEY` (or add `--api-key` to
`serve.sh`) or front the port with an authenticating proxy.

## Why this exists

Three ideas make a 335 GB model fit and fly on a 121 GB Spark:

1. **The 51.2B-parameter PLE n-gram table is quantized to NVFP4** (28.8 GB
   instead of 102.4 GB). A ~130-line loader patch teaches vLLM to read it.
2. **The table is served from NVMe via mmap** rather than kept resident. A
   token reads 16 rows × ~90 B, so the page cache does the work. This frees
   ~27 GB — which is what makes CUDA graphs and a 200K window fit at all.
   On unified memory it is the *only* offload that returns real memory (see
   the `VLLM_PLE_CPU_OFFLOAD` trap in [docs/REPRODUCE.md](docs/REPRODUCE.md)).
3. **Decode is bandwidth-bound, so the campaign moved fewer bytes**:
   weight-only FP8 (E4M3, per-channel, Marlin) for the dense projections and
   both lm_heads, CuteDSL skinny GEMMs for the tiny per-step matmuls, and two
   host-stall fixes. Each step now streams ~10.2 GB of weights at 76 % of the
   GB10's 246 GB/s pure-read ceiling.

| Checkpoint | Size | Engine |
|---|---|---|
| Official BF16 | 335 GB | any |
| Official FP8 | 173 GB | vLLM |
| Inferact NVFP4 | 182.8 GB | vLLM |
| RadixArk W4A4 | 135 GB | vLLM |
| Unsloth GGUF (dynamic, sub-4-bit) | 74–111 GB | llama.cpp only |
| **This checkpoint** | **109 GB** | **vLLM** |

## The patches (11, applied in lexical order)

| Patch | What it does |
|---|---|
| `10-sm121-marlin-thread-config` | fixes an sm121 Marlin thread-config bug (vllm#37030 class) that silently corrupts output |
| `20-ple-4bit-loader` | loads the packed NVFP4 PLE table (env `VLLM_PLE_NVFP4`) |
| `30-ple-graph-output-buffer` | makes the 4-bit PLE path CUDA-graph-capture-safe |
| `35-ple-nvfp4-mmap` | serves the PLE table from NVMe page cache (env `VLLM_PLE_NVFP4_MMAP`) |
| `40-mamba-eagle-drop` | prefix-cache crash with speculative decoding (from vllm#48375) |
| `41-mamba-state-seed` | prefix-cache crash on resume (from vllm#53142) |
| `50-jalapeno-fp8-proj` | weight-only FP8 Marlin for the three big dense projections (env `VLLM_JALAPENO_FP8_PROJ=proj`) — ×1.28 decode |
| `51-jalapeno-fp8-lmhead` | the same treatment for the main and draft lm_heads (`proj,lmhead`) — +12 % more |
| `52-jalapeno-skinny-gemm-gb10` | ships the model's CuteDSL skinny GEMM on GB10 for small-M matmuls (env `VLLM_JALAPENO_SKINNY`) |
| `53-jalapeno-async-attn-meta-h2d` | pinned async H2D for attention metadata — removes a 13.7 ms median host sync per decode step |
| `54-jalapeno-layer-name-cache` | memoizes torch LayerName opaques — 96 reconstructions per step eliminated |

All are Python-level over a digest-pinned vLLM base — no kernel rebuilds.
The Dockerfile *asserts* every patch landed (markers + compile checks), so a
half-applied patch fails the build instead of surfacing as garbled output at
inference time. Every Jalapeño patch is env-gated and inert when unset.

## Knobs

`serve/serve.sh` reads these from the environment (the compose file sets the
production values):

| Var | Default | Meaning |
|---|---|---|
| `PORT` | 8000 | listen port (compose uses 8006) |
| `CTX` | 131072 | context window; 204800 validated, 262144 native max |
| `SEQS` | 16 | max concurrent sequences |
| `GPU_MEM` | 0.78 | `--gpu-memory-utilization`; stay in 0.78–0.85 on GB10 |
| `MTP` | 2 | speculative tokens (0 off). 2 = production: +4 % single-stream, −1–2 % at c8–c16 |
| `CACHE` | 1 | prefix caching (requires patches 40+41, both in this image) |
| `MMAP` / `PREWARM` | 1 / 1 | PLE table from NVMe; prewarm fills the page cache at boot |
| `SLEEP_MODE` | 0 | expose `/sleep`, `/wake_up`, `/is_sleeping` (compose sets 1) |
| `FP8_PROJ` | proj,lmhead | Jalapeño patch 50/51 selection; empty disables |
| `SKINNY` | hc,router,ba,indexer | Jalapeño patch 52 GEMM families; empty disables |
| `ASYNC_H2D` | 1 | patch 53; 0 restores blocking copies |
| `LAYER_NAMES` | 1 | patch 54; 0 restores per-call construction |

## Operations

- **Update**: `git pull`, rebuild the image with a new tag, set `IMAGE=` in
  `.env`, `docker compose up -d`. Config changes are env-only — no rebuild.
- **Rollback**: point `IMAGE` at the previous tag (or flip the knob you
  changed) and `docker compose up -d` again. The Jalapeño patches are
  independently switchable via their env vars, so you can bisect a problem
  without rebuilding anything.
- **Sleep / share**: `curl -X POST localhost:8006/sleep` … `curl -X POST
  localhost:8006/wake_up`. Wake reloads weights from disk, so expect a
  multi-minute warm-up, not instant.

## GB10 platform notes

The traps that cost real time, condensed — the long versions with evidence
live in [docs/REPRODUCE.md](docs/REPRODUCE.md):

- **`GPU_MEM` above ~0.85 risks collapse**: GB10 reports reclaimable page
  cache as free memory, so vLLM's probe over-estimates (vllm#35313).
- **Benchmarks without `clocks.sm` logged are void** — GB10's DVFS parks
  bandwidth-bound decode well below max clocks on its own.
- **A second CUDA context cannot coexist with the resident engine** on
  121 GiB unified memory (host-OOM risk). Run kernel microbenches only with
  the engine stopped, or in CPU-only containers.
- FlashInfer's b12x CUTLASS MoE backend faults (Xid 31) on sm121 — Marlin is
  both the working and the faster path today.
- `--kv-cache-dtype fp8` is rejected by this model's QSA attention.
- Mamba/GDN prefix-cache bugs are geometry-dependent: uniform test prompts
  pass while varied geometry crashes. `serve/soak.py` exists for exactly this
  reason — use it after any cache or kernel change.

One honest caveat from weeks of production: our Spark showed intermittent
NVRM `NV_ERR_NO_MEMORY` bursts under heavy load. This class predates this
stack (it reproduced with an unmodified engine and other workloads), and the
serving config above was validated stable across the campaign — but if you
push the box hard, watch `dmesg`, and don't attribute it to the patches.

## Repository contents

- `docs/PERFORMANCE.md` — the Jalapeño campaign: promotion chain, decode
  anatomy, what failed and why, measurement methodology
- `docs/REPRODUCE.md` — end-to-end: quantize from the official BF16 release,
  build, serve, verify — every trap listed
- `docs/PREFIX-CACHING.md` — why prefix caching crashes GDN hybrids and how
  to test cache changes so they can't lie to you
- `patches/` — the eleven vLLM patches
- `serve/` — `serve.sh`, `gates.py` (acceptance gates), `soak.py` (seeded
  varied-geometry soak with killer-replay), `test_ple_mmap_parity.py`
  (offline bit-identity proof for the mmap path)
- `quantize/` — the incremental, resumable BF16 → NVFP4 pipeline (incl. the
  PLE table) and the loader-patch generator
- `calibrate/` — optional activation-max calibration for a checkpoint-level
  FP8 periphery
- `docker-compose.yml` + `.env.example` — the production serving stack
- `upload/` — Hugging Face upload tooling + the weights model card

## License and credit

Weights are a derivative of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
under the Qwen Community License 1.0 (included). Patches, Dockerfile, and
tooling: Apache-2.0.

Thanks to: Qwen (base model), RadixArk (calibration scales), Unsloth (the
4-bit n-gram precedent), namake-taro/vllm-custom (sm121 thread-config
insight), NVIDIA (sm121 vLLM enablement, upstream in 0.28).
