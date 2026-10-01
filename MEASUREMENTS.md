# Measurements: `DENSE=exl3` against `fp8` and `q4` (fork, 2026-10-01)

Two ASUS Ascent GX10 (GB10, 128 GB each), one QSFP56 DAC between the ConnectX-7 ports (one rail), GPU clocks not
capped. This recipe at v1.3 (TensorFold v0.6.0) with its defaults except `DENSE`: 4 streams, 1,048,576-token window,
FP8 KV cache, DFlash2 plus copy drafts, vision on. One boot per row, all rows the same day unless noted.

Measured before v1.3.1 (which sets `TF_GLM_MULTI_LONE=0` by default; its README puts the single-stream cost at
0.6-0.9%). The `exl3` change is independent of that setting.

## Decode, single stream: what agent work sees

`tools/decode_probe.py`: streaming, T=0, thinking off, time to first token excluded, median of 3.

| `DENSE` | Structured (count 1-200) | Prose (hash map, English) | **Code (French, binary search tree)** |
| --- | ---: | ---: | ---: |
| `fp8` | 102.5 | 51.7 | 69.6 |
| **`exl3`** | **109.4** | **59.9** | **77.6** |
| `q4` | 115.6 | 57.2 | 76.6 |
| *reference: vLLM nightly, EXL3 experts + EXL3 dense (our previous stack), same probe* | *90.6* | *41.6* | *54.1* |

The code prompt asks for a real program (a commented binary search tree, in French). Of the three probes it is the
closest to what an agent asks for. On it, `exl3` matches `q4` and is 11% above `fp8`.

## sparkDash

Same prompts for every row, through the OpenAI API, time to first token excluded. Read these with their prompts in
mind. "Structured" counts from 1 to 200. "Code" writes 50 copies of one `clamp_NN` function with only the suffix
changed, which favours the copy drafts. "Prose" is the hash-map explanation. So these rows compare settings, they
do not predict agent work.

| `DENSE` | 1 stream: prose / code / structured | 2 streams in all | 4 streams in all |
| --- | ---: | ---: | ---: |
| `fp8` | 51.4 / 116.1 / 99.7 | 70.4 / 151.9 / 137.3 | 95.8 / 248.4 / 165.1 |
| `exl3` | 66.8 / 122.9 / 107.1 | 80.6 / 195.1 / 111.1 | 104.8 / 257.3 / 221.2 |
| `q4` (v1.0, TensorFold 0.5.0) | 65.4 / 132.8 / 120.4 | 79.9 / 212.1 / 151.6 | 105.5 / 286.9 / 255.2 |

## Prefill and conversations

| | `fp8` | `exl3` (`TF_GLM_PREFILL_ROWS=4096`) |
| --- | ---: | ---: |
| 20k-token prompt, first time | 12.5 s | 11.7 s |
| 68k-token prompt, first time | 42.5 s | 39.9 s |
| Next turns of a 23k / 68k agent conversation | 0.4-0.8 s | 0.3-0.6 s |

At the recipe's default 2,048-row prompt chunks `exl3` fills ~9% slower than `fp8`. The prompt GEMM unpacks each
matrix once per chunk, and 4,096-row chunks amortize that over twice the rows.

## Quality

| | `bf16` | `fp8` | `exl3` | `q4` |
| --- | ---: | ---: | ---: | ---: |
| HumanEval, 164 problems × 5 samples, T=0.7, thinking off | 95.9% | 96.2% | 96.3% | 96.0% |
| French coding prompts (8 × 5 at T=0.7 + 8 at T=0) cut by `max_tokens` | 1 / 48 | 1 / 48 | 1 / 48 | **12 / 48** |
| P(end of turn) right after the closing code fence, French (`tools/end_of_turn.py`) | 0.76 | 0.82 | 0.88 | **0.55** |
| 8 French coding tasks run against their tests (T=0) | — | 8 / 8 | 8 / 8 | 8 / 8 |

The `bf16`, `fp8` and `q4` HumanEval and end-of-turn rows were measured on v1.0 (TensorFold 0.5.0, the same quantizers).
The `exl3` rows were measured on the exact v1.3 + `0054` + `0055` build. The differences in HumanEval are within the
95% bootstrap interval (±1.5 points). In the end-of-turn rows, `q4` stands apart.

## Memory

KV pool at start: `fp8` 2.32-2.57M tokens, `exl3` 2.05-2.14M (4,096-row chunks), `q4` 2.68-2.92M (it depends on what
is free when the server starts). Free memory at idle with `exl3`: ~7-9 GiB on each Spark.

## Reproduce

```bash
DENSE=exl3 ./start.sh restart
tools/decode_probe.py exl3
tools/end_of_turn.py exl3
```
