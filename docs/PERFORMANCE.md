# The Jalapeño performance campaign

How this stack went from 24.2 to ~38 accepted tok/s on a DGX Spark between
2026-09-02 and 2026-09-06 — every promotion measured, quality-gated, and
reversible, with the failures kept on the record so nobody repeats them.

The one-sentence summary: **decode on GB10 is weight-streaming-bound, and
the only lever that pays is moving fewer bytes per token** — plus removing
the host-side stalls that hid between GPU steps. Everything that didn't move
bytes or remove stalls was measured and rejected.

## The promotion chain

Anchor: `r6` = the recipe with patches 10–41 only, MTP=1 — already a fully
working server (mmap PLE, CUDA graphs, prefix caching), 24.2 ± 3.0 accepted
tok/s single-stream (A_interactive, the interactive benchmark arm).

| Promotion | Date | What | Patch / config | Measured gain | Cumulative |
|---|---|---|---|---|---|
| r7-e1 | 09-04 | weight-only FP8 (E4M3, per-channel) via Marlin for the three big dense projections | 50, `VLLM_JALAPENO_FP8_PROJ=proj` | ×1.278 → 30.9 tok/s | +28 % |
| r8-e1c | 09-05 | same treatment for main + draft lm_heads | 51, `proj,lmhead` | ×1.121 → 34.7 tok/s | +43 % |
| r9 | 09-05 | CuteDSL skinny GEMMs (small-M shapes) + two host-stall fixes | 52, 53, 54 | ×1.060 → ~36.8 tok/s | +52 % |
| r10 | 09-06 | MTP=2 speculative decoding (config-only) | `MTP=2` | ×1.042 → 37.97 ± 0.40 tok/s | **+56.9 %** |

Notes on the chain:

- **r9 was the best-predicted win of the project**: the bundle landed at
  99.3 % of its multiplicative prediction — evidence the performance model
  (below) was tracking reality by then.
- **MTP=2** was promoted only after a rollback and requalification: the
  first attempt was reverted within minutes by a pre-registered stability
  rule (a host-level driver fault, later shown to predate this stack), then
  re-promoted through the full gate chain once the host question was
  answered. Acceptance: 2.19–2.22 accepted tokens/step, p₂ ≈ 0.61–0.64,
  stable across concurrency.
- Each promotion shipped with a **2-line rollback diff** (image tag + env)
  preserved before the switch. None was ever needed for a quality reason.

## Where the time goes (r10 decode anatomy, single stream)

One decode step = verify pass (48 layers) + 2 draft passes. Per step the
engine streams **~10.2 GB of weights**:

| Class | GB/step | Share |
|---|---|---|
| NVFP4 activated experts (union of ~48–60 of 512/layer) | 3.86 | 38 % |
| FP8 dense projections (patches 50/51) | 2.73 | 27 % |
| Draft-head streams (×3 per step at MTP=2) | 1.27 | 12.6 % |
| BF16 remainder (latency-bound WMMA block, ~9 ms) | 1.31 | 13 % |
| Non-weight (GDN state, activations, KV, PLE rows) | ~0.6 | 6 % |

Aggregate read bandwidth: **187 GB/s = 76 % of the 246 GB/s pure-read
ceiling** of this GB10 — up from 72 % pre-r9. Per *accepted* token that is
~4.6 GB of weights. The remaining ~14 ms/step of non-bandwidth time is the
BF16 WMMA latency block (~9 ms) and residual host-sync work (~3–5 ms) — the
two known targets for anyone continuing this work (kernel fusion on the WMMA
block first; it is latency, not bytes, that binds it).

## Throughput by regime

Numbers to expect when you wire this into real workflows (metric vector, not
a single composite):

| Regime | Throughput | Notes |
|---|---|---|
| Single stream (interactive) | 38.7 tok/s accepted; ITL p50 56.8 ms | the number you feel in a chat UI |
| 4 streams | 21.2 tok/s aggregate | expert-union grows sublinearly |
| 8 streams | 14.6 tok/s aggregate | |
| 16 streams | 10.5 tok/s aggregate | expert streaming near saturation |
| Agent fanout (8 parallel CLI-agent conversations) | 23.8 tok/s aggregate, TTFT p90 6.06 s, per-stream p50 8.6 tok/s | the realistic agentic regime |
| Prefill | ~250–300 ms TTFT p50 short prompts; a 32K-token needle-retrieve answers in ~17 s | weight-streaming-bound, all 512 experts activate |

Decode draws ~32 W. Boot is ~6.5 min (198 s weight load via
`runai_streamer`; the default loader takes 674 s on GB10's pageable H2D
path). With prefix caching, conversation turns after the first drop from
~10 s to ~2–3 s of prefill at a 56–70 % hit rate.

## Quality evidence (why "fast" didn't mean "broken")

- **13/13 deterministic quality battery** at every promotion (math, retrieval
  to 32K context, translation, structured output).
- **Distributional checks**: logprob-identity legs (top-1 agreement 10/10,
  KL 0.0054–0.0121 vs the unpatched engine), sampled-temperature battery at
  n=975/arm judged non-inferior.
- The FP8 patches are **weight-only**: activations stay BF16, dispatch keeps
  a bf16-dequant path for prefill-size M (Marlin is ~2.9× slower than nvjet
  at large M), and the marlin-repacked + row-major FP8 storage is exactly
  memory-neutral vs BF16.

## What was tried and rejected

Published failures, with the reason, so the dead ends stay dead:

1. **W4A16 / NVFP4 for the remaining BF16 families** (router, HC, shared
   expert, ba, indexer): Marlin regresses to 0.33–0.85× at decode M=2 — these
   GEMMs are latency-bound with a ~19–21 µs kernel floor, so fewer bytes
   don't convert to less time. And weight-only 4-bit error is ~3.6× FP8's.
   Rejected on both axes, measured.
2. **FP8 KV cache**: this model's QSA attention requires a BF16 main KV cache
   and refuses to start.
3. **FlashInfer b12x CUTLASS MoE backend**: faults with Xid 31 during JIT
   warmup on sm121.
4. **`VLLM_PLE_CPU_OFFLOAD`**: frees nothing on unified memory — the "host"
   allocation comes out of the same 121 GiB pool. Only file-backed mmap
   returns real memory.
5. **A second CUDA context beside the resident engine** (for kernel
   microbenching): host-OOM risk on 121 GiB unified. Bench in stopped-engine
   windows or CPU-only containers only.
6. **Byte-equality quality gates**: vacuous for sampled configs — replaced by
   the powered battery above.
7. **Chasing the "kernel selection" story**: an early root-cause hypothesis
   falsified by re-profiling with correct warm-up handling. The discipline
   lesson, not the hypothesis, is what was worth keeping.

## How the numbers were measured

The campaign's rules, worth copying if you tune any of this:

- **Never trust a bandwidth number whose working set fits L2** (~25 MB) —
  rotate ≥256 MiB working sets.
- **3 reps cannot resolve <3 % deltas** — 8-rep measurement windows, rep-wise
  trend inspection, powered quality gates (n=975/arm) for accept/reject.
- **Log `clocks.sm` with every benchmark.** GB10 DVFS parks bandwidth-bound
  decode below max clocks; a stuck clock lock silently invalidates a day of
  numbers.
- **Pre-registered stop rules** decided *before* the window: e.g. MTP=2's
  first promotion rolled back inside 8 minutes on a stability rule that was
  written down hours earlier.
- **Investigate wins as aggressively as losses** — independent falsification
  before any KEEP; a win you can't explain is a measurement bug until proven
  otherwise.

A NOTE on scale: every number above is from one physical machine, stock
cooling, real workloads. Treat them as what to expect, not a guarantee;
the ratios between configurations are the transferable part.
