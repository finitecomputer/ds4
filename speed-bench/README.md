## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

To compare server-side decode configs through an already-running
OpenAI-compatible DS4 endpoint:

```
python3 speed-bench/http_decode_bench.py \
  --base-url http://127.0.0.1:8050/v1 \
  --model deepseek-v4-flash \
  --bearer-token-env BETA_INGRESS_TOKEN \
  --samples 3 \
  --json-out speed-bench/local-runs/ds4-dflash.json
```

The HTTP benchmark uses deterministic no-thinking, temperature-0 prompts and
reports per-prompt plus combined aggregate generation throughput.  Omit
`--bearer-token-env` for raw unauthenticated local endpoints.

### DS4-DFlash frontdoor experiments

The current DS4-DFlash work is measured against the same HTTP harness above.
Use the short decode prompts with `--warmups 1 --samples 2 --max-tokens 160`,
and use the sustained prompts with:

```
python3 speed-bench/http_decode_bench.py \
  --base-url http://127.0.0.1:8050/v1 \
  --model deepseek-v4-flash \
  --prompt prose512 \
  --prompt integer512 \
  --max-tokens 512 \
  --min-completion-tokens 500 \
  --warmups 1 \
  --samples 2
```

The important parser fix from the June 29 sweep is that speculators-format
DFlash configs can contain both a nested transformer `sliding_window` and a
top-level DFlash `sliding_window`.  The runtime must use the top-level DFlash
window.  For the current artifact that is `16`, not the nested transformer
window `2048`.

The best current DFlash candidate is opt-in, uses the sw16 artifact, and should
be treated as a long-output accelerator rather than a universal replacement:

```
DS4_DFLASH_BATCH_VERIFY=1
DS4_DFLASH_LAZY_PROMPT_HISTORY=1
DS4_DFLASH_DYNAMIC_DRAFT=1
DS4_DFLASH_DYNAMIC_DRAFT_START=1
DS4_DFLASH_DYNAMIC_DRAFT_MAX=2
DS4_DFLASH_DYNAMIC_DRAFT_GROW_EVERY=1
DS4_DFLASH_ADAPTIVE=1
DS4_DFLASH_ADAPTIVE_WINDOW=2
DS4_DFLASH_ADAPTIVE_MIN_ACCEPT_PCT=50
DS4_DFLASH_ADAPTIVE_COOLDOWN=64
DS4_DFLASH_PLAIN_FALLBACK=1
```

Launch with `--dflash-draft 2`.  Do not enable
`DS4_DFLASH_DRAFT_HIDDEN_HISTORY` for this artifact.  Feeding
DFlash draft hidden states back as target hidden history poisoned future
proposals in the June 29 diagnostic run.  With the config parser fixed, the
no-fallback accept-rate probe improved from roughly `1.6%` to `12.2%`; disabling
draft-hidden history improved the same probe to `34.5%`.  That is still too low
to keep retrying DFlash without adaptive fallback.

Latest same-time frontdoor numbers:

| config | prompt set | aggregate tok/s |
| --- | --- | ---: |
| no-DFlash DS4 | `prose160`, `integer128` | `15.0507` |
| DFlash sw16/no-hidden max2 W2/50 | `prose160`, `integer128` | `14.9703` |
| no-DFlash DS4 | `prose512`, `integer512` | `15.0716` |
| DFlash sw16/no-hidden max2 W2/50 | `prose512`, `integer512` | `15.1081` |

The sustained 512-token path is the real current win: `15.1081` tok/s versus
`15.0666` tok/s in the same-time no-DFlash baseline, about `+0.28%`.  The short
160-token path loses: `14.9703` tok/s versus `15.0521` tok/s, about `-0.54%`.

Other gates tried on June 29 did not beat the max2 long-output path:

| config | prompt set | aggregate tok/s | result |
| --- | --- | ---: | --- |
| DFlash max7 W2/50 | `prose160`, `integer128` | `15.0109` | worse than max2 and baseline |
| DFlash max7 W4/50 | `prose160`, `integer128` | `15.0184` | worse than baseline |
| DFlash max2 long-only>=256 | `prose160`, `integer128` | `15.0152` | reduces short loss but still below baseline |
| DFlash max2 long-only>=256 | `prose512`, `integer512` | `15.0941` | trims the sustained gain |
| DFlash max2 long-only>=161 | `prose512`, `integer512` | `15.0652` | effectively tied with baseline |
| Margin gate 0.75/cooldown16 | `prose160`, `integer128` | `14.1974` | rejected |
| Margin gate 1.00/cooldown16 | `prose160`, `integer128` | `14.2124` | rejected |
| Margin gate 1.50/cooldown16 | `prose160`, `integer128` | `14.2189` | rejected |
| Margin gate 1.00/cooldown32 | `prose160`, `integer128` | `14.1399` | rejected |

A live operator-frontdoor sanity run on June 29 used the already-exposed
`deepseek-v4-flash-ds4-dflash-test-fast` alias backed by the max2 sw16/no-hidden
server on `spark-2f73:8050`:

| frontdoor alias | prompt set | samples | aggregate tok/s |
| --- | --- | ---: | ---: |
| original DS4 fast | `prose512`, `integer512` | 1 each | `15.0652` |
| DFlash test fast max2 | `prose512`, `integer512` | 1 each | `15.1142` |
| original DS4 fast | `prose160`, `integer128` | 1 each | `14.7568` |
| DFlash test fast max2 | `prose160`, `integer128` | 1 each | `14.7946` |

Treat the frontdoor short-output pair as a smoke sample, not a routing decision;
the stronger five-sample raw short run still says not to route short/default
traffic to DFlash.  The frontdoor sustained sample agrees with the stronger raw
512-token result and confirms the separate Spark test alias can carry the
long-output DFlash win through the operator ingress.

After adding request-aware routing to the operator ingress, the original
`deepseek-v4-flash-q2-imatrix-ds4-fast` and
`deepseek-v4-flash-q2-imatrix-ds4-thinking` aliases keep short/default requests
on the original DS4 upstream and send explicit `max_tokens >= 384` requests to
the max2 DFlash upstream:

| original DS4 alias route | selected upstream |
| --- | --- |
| omitted `max_tokens` | `http://10.42.0.11:8000/v1` |
| `max_tokens=160` | `http://10.42.0.11:8000/v1` |
| `max_tokens=256` | `http://10.42.0.11:8000/v1` |
| `max_tokens=384` | `http://10.42.0.14:8050/v1` |
| `max_tokens=512` | `http://10.42.0.14:8050/v1` |

With the first 512-token routing deployed, the same original DS4 fast alias
measured:

| route | prompt set | samples | aggregate tok/s |
| --- | --- | ---: | ---: |
| original alias routed to DFlash | `prose512`, `integer512` | 2 each | `15.1140` |
| original alias staying on DS4 | `prose160`, `integer128` | 2 each | `14.7949` |

The 512-token live long-output number is faster than the previous original-DS4
frontdoor baseline (`15.0652` tok/s for the same 512-token smoke sample) while
preserving the original DS4 upstream for short/default requests.

The June 30 threshold probe then lowered the route threshold from 512 to 384.
Fast-path evidence:

| route | prompt set | samples | aggregate tok/s |
| --- | --- | ---: | ---: |
| original fast DS4, `max_tokens=384` before lowering | `prose512`, `integer512` | 1 each | `15.0320` |
| DFlash test fast, `max_tokens=384` | `prose512`, `integer512` | 1 each | `15.0711` |
| original fast alias routed to DFlash, `max_tokens=384` | `prose512`, `integer512` | 1 each | `15.0923` |
| original fast alias staying on DS4, `max_tokens=383` | `prose512`, `integer512` | 1 each | `15.0451` |
| original fast DS4, `max_tokens=256` before lowering | `prose512`, `integer512` | 1 each | `14.8999` |
| DFlash test fast, `max_tokens=256` | `prose512`, `integer512` | 1 each | `14.9259` |

Thinking-path evidence:

| route | prompt set | samples | aggregate tok/s |
| --- | --- | ---: | ---: |
| original thinking DS4, `max_tokens=384` before lowering | `prose512`, `integer512` | 1 each | `15.0333` |
| DFlash test thinking, `max_tokens=384` | `prose512`, `integer512` | 1 each | `15.1016` |
| original thinking alias routed to DFlash, `max_tokens=384` | `prose512`, `integer512` | 1 each | `15.0431` |
| original thinking alias staying on DS4, `max_tokens=383` | `prose512`, `integer512` | 1 each | `14.9374` |

The `256` probe remains too close and prompt-dependent to route by default.
Keep the threshold at 384 unless a wider A/B run contradicts the current
near-threshold controls.

So the corrected path should not replace the original DS4 frontdoor alias for
short completions.  The practical deployment shape is either a separate
DFlash-long test alias, or request-aware frontdoor routing where explicit
long-output requests go to a whole DS4-DFlash upstream on another Spark while
short/default requests stay on the original DS4 upstream.  Moving only the draft
work to another Spark should be treated as a separate sidecar experiment: it can
help only if measured draft execution is competing with target GPU work enough
to outweigh hidden-state transfer, network latency, and verifier wait time.

A current timing sample on the max2 adaptive runtime (`DS4_DFLASH_TIMING=1`,
`DS4_DFLASH_SPEC_LOG=1`, `DS4_DFLASH_VERIFY_TIMING=1`, and
`DS4_DFLASH_HISTORY_TIMING=1`) makes a remote draft sidecar look secondary, not
primary.  Timing flags themselves slow decode heavily, so use these as component
timings only:

| component | mean time |
| --- | ---: |
| DFlash proposal graph (`block + logits + select`) | `10.46 ms` |
| DFlash attempt draft side | `77.83 ms` |
| target hidden-history maintenance | `82.88 ms` |
| 2-token verifier top check | `108.81 ms` |
| DFlash attempt verifier side | `79.23 ms` |
| whole DFlash attempt | `157.05 ms` |

The sidecar implication is that simply running the small draft proposal graph on
another Spark cannot recover enough by itself; the target still pays hidden-tap
capture/history and verifier replay.  A sidecar is worth building only if it
lets us raise the draft cap while keeping acceptance high, or if the protocol
also removes a target-side history-maintenance cost.

Executor variants checked after the timing probe did not beat the max2 batch
verifier.  Discard the first long request after a server restart, or run a tiny
warmup, because the first prose request can be cold-start contaminated.

| config | prompt set | samples | aggregate tok/s | result |
| --- | --- | ---: | ---: | --- |
| max2 exact2 verifier, warm | `prose512`, `integer512`, `max_tokens=384` | 1 each | `13.2712` | rejected |
| max2 adaptive 75%, warm | `prose512`, `integer512`, `max_tokens=384` | 1 each | `15.0771` | no win over max2 W2/50 |
| max3 batch verifier, warm | `prose512`, `integer512`, `max_tokens=384` | 1 each | `13.4671` | rejected |
| max2 cooldown-only adaptive, warm | `prose512`, `integer512`, `max_tokens=384` | 1 each | `14.1936` | rejected |
| max2 dynamic start=2, warm | `prose512`, `integer512`, `max_tokens=384` | 1 each | `15.0499` | no win |
| max2 sparse real history, warm | `prose512`, `integer512`, `max_tokens=384` | 1 each | `15.0521` | no win |
| delayed accepted-row logits readback patch, warm | `prose512`, `integer512`, `max_tokens=384` | 2 each | `15.0108` | rejected and rolled back on Spark |
| hidden-history max3 adaptive80, frontdoor | `prose512`, `integer512`, `max_tokens=512` | 1 each | `15.0389` | rejected |
| noncausal artifact max2, frontdoor | `prose512`, `integer512`, `max_tokens=384` | 1 each | `15.0106` | rejected |
| max2 bounded plain-fallback retry64, frontdoor | `prose512`, `integer512`, `max_tokens=384` | 1 each | `14.8736` | rejected |
| max2 adaptive window1, frontdoor | `prose512`, `integer512`, `max_tokens=384` | 1 each | `14.9783` | rejected |
| max2 second-draft margin cap 1.4, frontdoor | `prose512`, `integer512`, `max_tokens=384` | 1 each | `15.0582` | no win |
| max2 second-draft margin cap 1.0, frontdoor | `prose512`, `integer512`, `max_tokens=384` | 1 each | `14.9753` | rejected |
| max2 target-margin gate 8, frontdoor | `prose512`, `integer512`, `max_tokens=384` | 1 each | `15.0481` | rejected |
| max2 target-margin gate 4, frontdoor | `prose512`, `integer512`, `max_tokens=384` | 1 each | `15.0582` | no win |
| max2 pre-margin gate 8, frontdoor | `prose512`, `integer512`, `max_tokens=384` | 1 each | `15.0394` | rejected |
| max2 disable prefix1 capture, frontdoor | `prose512`, `integer512`, `max_tokens=384` | 1 each | `15.0644` | no win |
| max2 tail gate min-remaining 64, frontdoor | `prose512`, `integer512`, `max_tokens=384` | 1 each | `15.0595` | no win |

An accept-log probe of the max2 batch adaptive config over one warmed
`max_tokens=384` frontdoor run showed only `9` DFlash attempts across the
sampled requests after fallback behavior took over.  That explains why the
current win is small and why cooldown-only adaptive is worse: the runtime is
mostly protecting baseline DS4 throughput, not sustaining high DFlash
acceptance.  Prior alignment probes already rejected row-base `0`, position
offset `+1`, and tap-layer offset `+/-1`; the sw16/no-hidden interpretation
remains the best available read of the artifact.

The delayed-readback patch moved the full-vocab verifier logits read from the
batch verifier helper to the accepted-prefix path, avoiding one wasted row read
on prefix1 partial accepts.  It did not improve the warmed frontdoor aggregate,
which suggests the verifier layer replay and acceptance rate dominate more than
this readback cost.

The noncausal artifact had a better earlier no-fallback accept probe than sw16,
but that did not translate into frontdoor throughput; verifier and attempt costs
still dominated.  A bounded plain-fallback retry was also worse than permanent
plain fallback, which means the current adaptive behavior is correctly avoiding
later low-value attempts rather than missing a large easy suffix.

Warm history-maintenance timing also ruled out allocation reuse as the next
useful optimization: fast target-token history allocation was about `0.002 ms`
median, while target eval/tap capture was about `64 ms` and projection about
`3.3 ms` after cold start.  A second-draft margin cap tried to avoid likely
bad 2-token verifier passes, but thresholds `1.4` and `1.0` did not beat the
current max2 adaptive config.

A confidence-labeled no-fallback diagnostic run collected `309` accept-log rows.
The confidence signals were real: index-0 hits had median draft margin `2.83`
versus `0.64` for misses, median target margin `14.27` versus `1.93`, and
median pre margin `16.46` versus `1.69`.  Offline target/pre thresholds raised
the kept-row accept rate above `80%`, but real frontdoor throughput did not
improve.  Target-margin gates at `8` and `4`, plus a pre-margin gate at `8`,
all stayed below the current max2 adaptive evidence.  So the current bottleneck
is not merely detecting bad attempts; skipping them also discards enough useful
accepted suffixes, or disrupts DFlash history enough, that aggregate throughput
does not improve.

Disabling prefix-1 capture tested the MTP-style verifier tradeoff: avoid copying
prefix-1 compressor frontiers on full accepts, but replay or restore more work
on partial accepts.  It reached `15.0644` tok/s on the `max_tokens=384`
frontdoor probe, which is close but still below the best current max2 evidence.
Keep prefix-1 capture enabled for DFlash.

A tail gate with `DS4_DFLASH_MIN_REMAINING_TOKENS=64` also stayed below the
current max2 evidence at `15.0595` tok/s.  Tail gating does not recover enough
bad-attempt cost to justify changing the live recipe.

After these checks, the live DFlash test slot was restored to the max2 batch
adaptive config above on `spark-2f73:8050`.

## DSpark executor checkpoint

The next executor path is DSpark, not a remote draft sidecar yet.  The MTP-only
DSpark payload is staged on `spark-cbee` at
`/home/finite/ds4-dspark/fraserprice-DeepSeek-V4-Flash-DSpark`:

| shard | size | role |
| --- | ---: | --- |
| `dspark-mtp-00001-of-00003.safetensors` | `3,610,455,184` bytes | `mtp.0` |
| `dspark-mtp-00002-of-00003.safetensors` | `3,560,111,960` bytes | `mtp.1` |
| `dspark-mtp-00003-of-00003.safetensors` | `3,692,775,244` bytes | `mtp.2` |

The dependency-free inspector validates the complete sharded layout and the
critical executor tensors:

| tensor | dtype | shape | shard |
| --- | --- | ---: | --- |
| `mtp.0.main_proj.weight` | `F8_E4M3` | `[4096, 12288]` | `00001` |
| `mtp.0.main_proj.scale` | `F8_E8M0` | `[32, 96]` | `00001` |
| `mtp.0.main_norm.weight` | `BF16` | `[4096]` | `00001` |
| `mtp.2.norm.weight` | `BF16` | `[4096]` | `00003` |
| `mtp.2.markov_head.markov_w1.weight` | `BF16` | `[129280, 256]` | `00003` |
| `mtp.2.markov_head.markov_w2.weight` | `BF16` | `[129280, 256]` | `00003` |
| `mtp.2.confidence_head.proj.weight` | `BF16` | `[1, 4352]` | `00003` |
| `mtp.2.hc_head_fn` | `F32` | `[4, 16384]` | `00003` |
| `mtp.2.hc_head_base` | `F32` | `[4]` | `00003` |
| `mtp.2.hc_head_scale` | `F32` | `[1]` | `00003` |

The current C executor work has crossed the first real artifact boundary:
`ds4_dspark_config` loads and validates the DSpark config, `ds4_dspark_weights`
maps multiple MTP shards through `model.safetensors.index.json`, binds the FP8,
BF16, and F32 bootstrap tensors above, and reads BF16 tensors from the owning
shard.  `tests/ds4_dflash_config_test.c` includes a tiny two-shard DSpark
fixture that exercises the real split between `mtp.0` and `mtp.2`.  The C path
also has reference implementations for the first real DSpark executor pieces:
`mtp.0.main_proj + main_norm`, HC collapse/expand, FP8 linears, and BF16
RMSNorm.

The DS4 frontdoor can now recognize a DSpark artifact through the existing
`--dflash` option: it first tries the classic DFlash loader, then falls back to
`ds4_dspark_config` plus `ds4_dspark_weights_open_graph`.  In `--inspect-only`
mode this validates and binds the DSpark graph payload as a first-class engine
artifact.  Non-inspect runtime remains intentionally gated with
`DS4_DSPARK_EXPERIMENTAL_RUN=1`.

The Spark-side C probe now confirms this path against the real staged artifact
and uses `ds4_dspark_weights_open_graph`, which binds every MTP stage tensor:
HC parameters, attention projections, gate tensors, shared experts, and all
256 routed FP4 experts for each of `mtp.0`, `mtp.1`, and `mtp.2`.

```text
cd /home/finite/ds4-dspark/c-probe
cc -O2 -Wall -Wextra -std=c99 -I. -o dspark_loader_probe \
  dspark_loader_probe.c ds4_dflash.c -lm -pthread
./dspark_loader_probe /home/finite/ds4-dspark/fraserprice-DeepSeek-V4-Flash-DSpark
```

For live-path timing, skip the older single-row diagnostics and the full-vocab
Markov diagnostic sweep:

```text
DS4_DSPARK_BLOCK_TIMING=1 \
DS4_DSPARK_THREADS=auto \
DS4_DSPARK_PROBE_SKIP_SINGLE=1 \
DS4_DSPARK_PROBE_SKIP_MARKOV=1 \
./dspark_loader_probe /home/finite/ds4-dspark/fraserprice-DeepSeek-V4-Flash-DSpark
```

Result on `spark-cbee` after adding the row-coupled block-stage reference path
and updating the probe to run both the legacy single-row stage path and the
5-row block path:

```text
real 43.94
user 43.84
sys 0.04
dspark probe: loaded 3 shards, bound 4705 tensors
dspark probe: block=5 hidden=4096 target_layers=3 markov_rank=256 sliding_window=128 stages=3
dspark probe: main_x checksum=-60.712188954 first=[-0.084533282 -0.035384785 -0.030015269 -0.053174287]
dspark probe: mtp0.stage rope_pos=17 attn_hc_checksum=-526.372671437 ffn_hc_checksum=945.465437351 ffn_out_checksum=574.034185166 routed_sum_checksum=285.263387576 shared_out_checksum=288.770814532 routes=[203:0.323420912 108:0.273852289 150:0.223186970 151:0.221319124 59:0.211899638 101:0.246321067] first=[5.639304638 -4.812329292 -1.948553562 -0.705649853]
dspark probe: mtp1.stage rope_pos=18 attn_hc_checksum=-16416.264616180 ffn_hc_checksum=-16817.217999344 ffn_out_checksum=-108.733477585 routed_sum_checksum=-393.378274193 shared_out_checksum=284.644773717 routes=[232:0.280398458 59:0.268550456 134:0.252567917 105:0.241083682 115:0.232855290 7:0.224544212] first=[-1.185502410 -5.246861458 4.740826130 1.247539163]
dspark probe: mtp2.stage rope_pos=19 attn_hc_checksum=-6253.304233603 ffn_hc_checksum=-5112.803708339 ffn_out_checksum=-760.203074090 routed_sum_checksum=-411.236844689 shared_out_checksum=-348.966186567 routes=[249:0.300106406 196:0.297479510 86:0.228599086 237:0.226686478 233:0.223732740 67:0.223395795] first=[-11.455233574 2.053888321 1.906604052 -0.425103873]
dspark probe: mtp0.block rows=5 checksum=13597.970807212 row0=[7.696265221 -11.208616257 -3.535907269 -0.983807504] row4=[8.275209427 -10.436101913 -1.406228900 0.220118582]
dspark probe: mtp1.block rows=5 checksum=-39397.813655565 row0=[-2.607398033 -12.670503616 2.416069508 -1.794953227] row4=[-2.311804771 -12.667621613 4.892149925 0.012597888]
dspark probe: mtp2.block rows=5 checksum=-102099.719354188 row0=[4.154207230 -7.619506836 -8.033886909 1.558397055] row4=[5.082611084 -6.729090691 -6.116759777 3.990957499]
dspark probe: final_hc checksum=-296.241743947 norm_checksum=-106.218600657 first=[-0.019129066 -0.008259543 -0.154023737 -0.054884553]
dspark probe: markov prev_token=42 embedding_checksum=75.634155273 logits_checksum=-8323370.127847247 top=6002 top_logit=11.456508636 confidence_logit=4.327947140
dspark probe: block_final rows=5 hidden_checksum=-2960.798638670 norm_checksum=-222.024593119 row0=[0.409557492 -0.291963726 -0.234342590 -0.015782310] row4=[0.410379529 -0.254553258 -0.164444730 0.050164185]
```

After replacing repeated FP8/FP4 decode calls with once-initialized lookup
tables, the same checksums are preserved but the live block runner is much
faster.  The full probe still spends about `30s` opening/binding the staged
artifact and running optional diagnostics, so the hot-path number to watch is
the per-stage timing:

```text
ds4: dspark block timing stage=0 rows=5 bind=0.004 ms alloc=0.162 ms qkv=102.683 ms sparse_attn=1.249 ms attn_out=172.050 ms ffn=459.893 ms total=736.041 ms
ds4: dspark block timing stage=1 rows=5 bind=0.069 ms alloc=0.030 ms qkv=103.739 ms sparse_attn=1.102 ms attn_out=173.065 ms ffn=460.769 ms total=738.774 ms
ds4: dspark block timing stage=2 rows=5 bind=0.131 ms alloc=0.026 ms qkv=103.719 ms sparse_attn=1.126 ms attn_out=173.958 ms ffn=461.446 ms total=740.406 ms
```

That moves the three-stage 5-row DSpark block from roughly `13.6s` to roughly
`2.22s`.  Sparse attention is not the bottleneck; FP8/FP4 projections and the
FFN/MoE branch still dominate.

After adding optional row-parallel projection loops for the DSpark FP8, grouped
FP8, BF16, and FP4 linears, set `DS4_DSPARK_THREADS=auto` or an explicit thread
count to use the host CPU.  The default remains single-threaded.  On
`spark-cbee`, the same hot block timing with `DS4_DSPARK_THREADS=20` is:

```text
ds4: dspark block timing stage=0 rows=5 bind=0.004 ms alloc=0.167 ms qkv=15.821 ms sparse_attn=0.500 ms attn_out=20.200 ms ffn=82.504 ms total=119.196 ms
ds4: dspark block timing stage=1 rows=5 bind=0.066 ms alloc=0.030 ms qkv=15.949 ms sparse_attn=0.335 ms attn_out=20.058 ms ffn=83.044 ms total=119.482 ms
ds4: dspark block timing stage=2 rows=5 bind=0.127 ms alloc=0.029 ms qkv=15.693 ms sparse_attn=0.333 ms attn_out=20.345 ms ffn=83.684 ms total=120.211 ms
```

The checksums match the single-threaded control build for the same compiler
flags, so the parallel path changes scheduling, not math order within an output
row.  Thread-count tuning on the same host:

| `DS4_DSPARK_THREADS` | stage total range |
| ---: | ---: |
| `6` | `293.224-294.200 ms` |
| `10` | `190.193-192.416 ms` |
| `14` | `149.296-149.859 ms` |
| `20` / `auto` | `119.196-120.211 ms` |

That moves the hot three-stage 5-row DSpark block from roughly `2.22s` to
roughly `0.36s` after the decode-table optimization, and from roughly `13.6s` to
roughly `0.36s` versus the first scalar decoder.  The bottleneck is still
FFN/MoE projection work, just no longer catastrophically single-threaded.

`DS4_DSPARK_F32_ACCUM=1` is an experimental speed/acceptance probe that switches
the DSpark linears from conservative double accumulation to FP32 accumulation.
On `spark-cbee` with `DS4_DSPARK_THREADS=20`, it measured `104.581 ms`,
`108.094 ms`, and `117.321 ms` for the three stages, but it changes draft
checksums.  Do not treat it as a default until live verifier acceptance is
measured.

The `4705` bound tensors are exactly the full DSpark MTP payload:
`10` bootstrap/final tensors plus `3 * (29` stage-core tensors `+ 256 * 6`
routed-expert tensors`)`.

The important architecture catch remains: the DSpark MTP shards do not include
`embed.weight`, `head.weight`, or the shared base `norm`/HC-head tensors.  DS4
must reuse the target embedding/output projection for those pieces.  DSpark has
its own `mtp.2.norm.weight`, so the integration must not double-normalize via
the target plain-output helper.  Use
`ds4_session_eval_output_projection_from_normed_plain` after `mtp.2.hc_head` and
`mtp.2.norm`, then add the Markov bias before token selection.

The probe now follows the true stage topology through captured-tap projection,
HC-stream initialization, stage HC attention collapse, attention projections,
partial RoPE with YaRN frequency smoothing, single-KV sparse attention with the
learned attention sink, inverse output RoPE, grouped output projection, attention
HC expansion, FFN-side HC collapse, router top-k, the shared FP8 SwiGLU expert
branch, all six selected routed FP4 experts, and FFN HC expansion.  The runner
then applies `mtp.2.hc_head`, `mtp.2.norm`, the vanilla Markov W1/W2 bias, and
the confidence projection over `[hidden, prev_markov_embedding]`.  The block
runner now also runs `block_size` HC rows through all MTP stages with row-coupled
draft attention over the block rows.  It is still a CPU reference path and keeps
the current sparse-attention approximation, so `43.94s` is an optimization
warning, not a deployable speed number.

`ds4_dspark_final_block_norm_f32` now finalizes a full DSpark block by applying
`mtp.2.hc_head` and `mtp.2.norm` row-by-row, and the probe uses that shared
helper instead of an inline one-off loop.  The remote artifact output above is
from the same math before this helper extraction; the 2026-06-29 rerun was not
completed because Tailscale SSH stayed at the extra auth check.  Local coverage
does exercise the helper against the tiny two-shard fixture, and the standalone
probe still compiles against the helper.

`ds4_dspark_select_draft_tokens_argmax` applies the Markov correction and
confidence gate to already-projected DSpark base-logit rows, producing compact
draft token ids, margins, and optional confidence logits.  `ds4.c` also has an
internal `ds4_session_dspark_project_select_argmax` bridge that feeds finalized
DSpark normed rows through the target output projection and then calls that
selector.  The live DSpark speculative path now accepts the first target token
through a tapped graph decode, collapses the captured HC taps through the target
output HC-head into DSpark `main_hidden`, runs `ds4_dspark_project_main_hidden`,
initializes the DSpark HC block from the anchor projection, runs every MTP stage
with the block runner, finalizes the block, selects compact target-vocab draft
ids, and verifies them with the existing target batch-verifier pattern when
possible.  `DS4_DSPARK_SPEC_LOG=1` and `DS4_DSPARK_TIMING=1` expose draft,
acceptance, and timing diagnostics.  `DS4_DSPARK_SPEC_DISABLE=1` loads DSpark
without using it for speculative decode, and
`DS4_DSPARK_CONFIDENCE_THRESHOLD` can trim low-confidence proposal suffixes.
`DS4_DSPARK_DISABLE_MARKOV=1` is a verifier-safe speed experiment that skips the
full-vocab Markov correction during draft selection and chooses from DSpark base
logits only.  This may reduce acceptance, but it cannot corrupt the target token
stream because the target verifier still accepts or rejects every draft token.
`DS4_DSPARK_TAP_COLLAPSE=output-hc|stream0|mean` selects how captured target HC
taps are reduced before `main_proj`; the default is `output-hc`, while
`stream0` and `mean` are live acceptance probes for the still-unproven tap
convention.  The block runner now binds fixed MTP-stage tensors once per stage,
uses decode lookup tables for FP8/FP4/BF16 scalar reference math, and proposal
setup uses `ds4_dspark_init_hc_block_from_main_f32` so the live path and artifact
probe initialize HC rows identically before any synthetic probe jitter is added.

The next proof point is no longer "can DSpark produce proposals at all"; it is
whether the current tap convention is the right one and whether accepted
proposal length beats the added CPU reference draft cost.  The current live path
uses the target output HC-head to collapse intermediate HC taps to plain hidden
rows because the DSpark MTP shards expect `hidden_size * n_target_layer_ids`
input, while the DS4 graph captures HC-width rows.  If acceptance is poor, this
HC-to-plain convention is the first suspect.  If acceptance is good but speed is
poor, the next optimization is moving the DSpark proposal graph off the CPU
reference path and/or onto a separate draft Spark that sends only compact token
ids and confidences back to the target/verifier Spark.
