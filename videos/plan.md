# Video series plan

Audience: engineers who already know transformers. Length is flexible, so each video goes as deep as its sources allow.

Every beat below comes from a section of this repo. The "Gaps" line in each video lists material the repo doesn't cover, which would have to come from outside sources (marked so it can be cited separately).

Visual motifs that recur across videos:

- **Memory bar** (bytes per parameter) - introduced in #5, sliced across GPUs in #7, shrunk to weights + KV cache in #11
- **Timeline lanes** (compute / comms / DataLoader) - #6, #7, #10
- **Furnace** (the locomotive fireman) - the opener in #1, revisited whenever something "starves the GPU"

Arc: one GPU (1-5) → many GPUs (6-10) → serving (11) → running for real (12-13).

---

## 1. Feeding the furnace

**Thesis:** the bottleneck in ML hardware is moving bits, not computing on them.

Beats:

1. The steam locomotive: a great engine is useless if the fireman can't shovel coal fast enough.
2. Compute outgrows the pipes. NVLink generations vs fp16 TFLOPS: V100 → Rubin is 32x compute but only 12x intra-node bandwidth.
3. A transformer has three kinds of ops (from "Data Movement Is All You Need"): tensor contractions (matmuls), statistical normalizations (softmax, layernorm), element-wise ops (bias, dropout, activations, residuals). Only the first group is compute-heavy.
4. Where bits move: HBM ↔ SMs, GPU ↔ GPU in a node, node ↔ node, storage → DataLoader. Each one can starve the GPU.
5. The fix for HBM traffic is fusion: flash attention, `torch.compile`. Eager mode pays an HBM round trip per op; a fused kernel pays it once per chain.
6. Payoff: the same `torch.compile` on a B200 gives up to ~6x on Llama-3.2-1B at seq 512 (launch/bandwidth bound) and 0.94x on Llama-3.1-8B at seq 8192 (matmul bound). Same tool, different bottleneck.

Visuals: locomotive; two diverging growth curves (compute vs NVLink); ops colored by what bounds them; the two compile tables as bars.

Sources: `insights/ai-battlefield.md` (Can you feed the furnace fast enough?, Moving bits, Network table), `training/performance/README.md` (Anatomy of Model's Operations, torch.compile → Where it pays off).

Gaps: the repo never draws a roofline or defines arithmetic intensity. A roofline would need to be derived (e.g. H100 ridge point = 989 TFLOPS / 3.35 TBps ≈ 295 FLOP/byte). Mark as added material.

---

## 2. Counting FLOPs

**Thesis:** you can predict training time from arithmetic, but only if you know which "TFLOPS" you're using.

Beats:

1. MAC vs FLOP: `a*b+c` = 2 FLOPs. A `(m×k)@(k×n)` matmul is `m·n·k` MACs = `2mnk` FLOPs. FLOP vs FLOPS vs FLOP/s naming mess.
2. Where the spec number comes from: `clock × FMAs_per_clock_per_unit × 2 × units`. H100: 1830MHz × 512 × 2 × 528 = 989 TFLOPS. Inverting the formula exposes the real clock (the widely quoted 1980MHz would give 1070).
3. FLOPs per training step: forward ≈ 2 per param per token, backward 2x forward (grads wrt inputs and weights) → factor 3; activation recompute adds a forward → factor 4. Formula: `model_size_in_B * 4 * 2 * seqlen * gbs / (time * gpus * 1e3)`.
4. Three ceilings: theoretical peak vs MAMF (short boost-clock burst) vs MSMF (sustained at power limit). H200 bf16: 989 / 834 / 755. Why: a saturated matmul hits TDP and the clock drops (B200: 2250 × 1425/1965 ≈ 1632 vs 1429 measured).
5. MFU vs HFU: MFU counts only model FLOPs, HFU also counts recompute. Megatron table (175B: 51.4% MFU, 52.8% HFU). BLOOM on A100: 150 of 312 TFLOPS was "amazing".
6. Turning it into money: total FLOP / achieved TFLOPS = days. 7 days at peak becomes 14 days at 50% MFU.
7. Why MFU numbers across papers don't compare: different FLOP counting, different timing windows.
8. Silicon lottery: 5%+ spread on one node; the slowest GPU sets the pace.

Visuals: matmul as a cube of MACs; the spec formula as three dials; three bars (peak / MAMF / MSMF); clock and power trace during a burst vs sustained run.

Sources: `training/performance/README.md` (MACs vs FLOP vs FLOPS, TFLOPS as a performance metric, MFU vs HFU), `compute/accelerator/README.md` (How To Calculate Theoretical TFLOPS, MAMF/MSMF tables, Why MSMF is well below Theory, Not all accelerators are created equal), `insights/ai-battlefield.md` (TFLOPS, MFU).

---

## 3. Tiles and waves

**Thesis:** a GPU computes in discrete chunks, so matrix shapes that look "almost the same" can run 2x apart.

Beats:

1. A GEMM's output is cut into tiles; each tile is a thread block scheduled onto an SM (A100: 108 SMs, H100: 132).
2. Tile quantization: if a dimension doesn't divide the tile, the last tile does full work for partial output. Tensor-core alignment: multiples of 8 elements on V100 fp16, 64 on A100.
3. Wave quantization: 109 blocks on 108 SMs = 2 waves, the second almost empty but costing almost a full wave.
4. Do more SMs give more TFLOPS? `speedup = ceil(C/S1)/ceil(C/S2)` is a staircase. B300 with 148 vs 152 SMs (2.7% more): 1.000x at 148 blocks, 2.000x at 152, 1.000x at 296, 1.042x for a real Llama-3.1-8B MLP shape (3584 blocks), 1.027x asymptote.
5. Published peak is an aggregate over the whole part, so one name can't honestly carry one number across SKUs (2250 vs 2311 bf16).
6. SwiGLU's 8/3 trap: `d_ff = 8/3 × 4096 = 10922` runs at 273 TFLOPS on H100; 10944 (22 more) runs at 398 (+46%). Llama-2-7B picked 11008.
7. Keep `h/a` large; flash attention removes most head-sizing constraints.
8. Sizing checklist: vocab divisible by 64, max micro-batch, `b*s`, `h/a`, `h/t` divisible by powers of 2, `(b*a)/t` integer, small `t`, search near `8/3*h`.

Visuals: tiles falling into SM slots wave by wave; the staircase plot; SwiGLU sweep as bars.

Sources: `training/performance/README.md` (Vector and matrix size divisibility → Final recommendations for model sizing), `compute/accelerator/README.md` (TFLOPS vs matrix size chart).

---

## 4. Bits of a number

**Thesis:** every bit you drop buys speed, and each format decides where to spend the bits it has left.

Beats:

1. Anatomy: sign / exponent / mantissa. Exponent = range, mantissa = precision. fp32, tf32 (19 bits), fp16, bf16, fp8 E4M3 / E5M2.
2. The progression: fp32 → fp16 mixed precision (fast but unstable, max ~64k) → bf16 (fp32's range, same protocol) → fp8 (DeepSeek-V3 trained in it) → fp6/fp4 on Blackwell.
3. Each halving ≈ 2x throughput (B200: fp32 80, tf32 1125, bf16 2250, fp8 4500) until fp6, which runs at fp8 speed. fp4 doubles again (3x on GB300).
4. TF32: 8x fp32 on A100 (19.5 → 156), off by default.
5. Where precision must stay high: accumulation. Reductions, gradient accumulation, the optimizer step (a tiny gradient added to a big weight disappears) → fp32 master weights and states, or Kahan summation / stochastic rounding.
6. fp4 e2m1 has 16 bit patterns (±0, 0.5, 1, 1.5, 2, 3, 4, 6), so it only works block-scaled. mxfp4: 32-element blocks, power-of-two E8M0 scale, 4.25 bits/element. nvfp4: 16-element blocks, E4M3 scale + fp32 tensor scale, 4.5 bits. An outlier distorts 15 neighbors instead of 31. NVIDIA reports MXFP4 needed ~36% more tokens to match NVFP4 loss.
7. Reading a dtype name: `float8_e4m3b11fnuz`.
8. Switching after training: bf16-trained → fp16 overflows; fp16-trained → bf16 usually works.

Visuals: bit-field strips side by side; a number line showing representable values (fp4's 16 dots); a block scale stretching those dots over a block's range; a small number vanishing when added to a large one.

Sources: `training/dtype.md`, `inference/README.md` (Model Weights, MX byte costs), `training/instabilities/README.md` (Numerical instabilities).

Gaps: fp16 loss scaling is only mentioned in passing.

---

## 5. Where does memory go?

**Thesis:** a model's weights are a small part of what training puts on the GPU.

Beats:

1. 18 bytes per parameter in mixed precision with AdamW: weights 6 (fp32 master + bf16 copy), grads 4, Adam states 8.
2. Cheaper optimizers: 8-bit Adam (2 bytes), Adafactor / LION (4), all-bf16 AnyPrecisionAdamW (8 bytes total for weights + states + grads).
3. Activations, worked on Llama-3.1-8B at bs=1, seq=32K:
   - one `hidden_states` tensor = 0.25GiB
   - one layer's forward makes ~28 copies → 7GiB (varies: SmolLM2 24, Gemma 48)
   - 32 layers without checkpointing → 224GiB; with checkpointing → 8GiB
   - fp32 logits = 4 × 32768 × 128256 → 15.7GiB, more than all the checkpoints combined
   - total: 31GiB with checkpointing vs 240GiB without
4. Gradient checkpointing: ~20-25% slower per step (papers: up to 30-40%), but the freed memory buys a bigger batch, so it's faster overall.
5. Memory you never asked for: CUDA kernels 0.5-2GiB, `torch.distributed` 1-2GiB (nccl2: 5.16GiB), fragmentation (`expandable_segments:True`).
6. Batch sizes: MBS, GBS = MBS × DP × GAS. Gradient accumulation trades steps for memory and cuts DDP comms by GAS.
7. Instant math: training needs `params_B × 18 × 1.25 / GPU_GB` GPUs; inference `params_B × 2 × 1.25 / GPU_GB`. 80B on 80GB GPUs: 23 for training, 3 for inference.

Visuals: the stacked memory bar (introduced here); an activation tower per layer collapsing under checkpointing; the logits slab towering over it.

Sources: `training/performance/README.md` (Anatomy of Model's Memory Usage, Additional GPU memory usage, Batch sizes, Gradient Accumulation, Gradient Checkpointing, Memory-efficient optimizers), `insights/ai-battlefield.md` (Tell how many GPUs do you need in 5 secs).

---

## 6. Collectives

**Thesis:** distributed training speaks a small vocabulary of communication patterns, and their costs are predictable.

Beats:

1. Point-to-point (`send`/`recv`, used by PP) vs collectives vs one-sided (nccl2 windows, for sparse access like routing a token to one expert).
2. The vocabulary, each with its user: broadcast, gather, all-gather (ZeRO weight gathering), reduce, all-reduce (DDP grads), scatter, reduce-scatter (ZeRO grads), all-to-all (Ulysses SP, MoE EP).
3. all-reduce = reduce-scatter + all-gather. That's why all-reduce moves `2(n-1)/n` × payload and reduce-scatter moves half.
4. Ring broadcast: split into `S` chunks so all links work at once. Time `N(S+k-2)/(SB)` → `N/B` when `S >> k`.
5. Tree: hops grow as `log(k)`, not `k`. NCCL's double binary tree keeps every node's outbound bandwidth busy. Ring wins on bandwidth, tree on latency.
6. Protocols: LL (4B data + 4B flag, ~1µs/hop, 25-50% of peak), LL128 (120B + 8B flag, ~95%, NVLink only), Simple (memory fences, near peak, ~6µs).
7. algbw vs busbw: busbw scales by the collective's correction factor so it reflects the wire, not the rank count.
8. SHARP: the switch does the reduction, N+1 sends instead of 2N. H200 all-reduce busbw 482GBps against a 450GBps spec. Kicks in above 4 GPUs on H200, above 5 on B200, and only for all-reduce.

Visuals: 4 GPUs as colored blocks for each collective; chunks pipelining around a ring; the double tree; a switch doing arithmetic.

Sources: `network/comms.md`, `network/README.md` (Glossary algbw/busbw, SHARP), `training/model-parallelism/README.md` (Parallelism network collectives).

---

## 7. Data parallelism and ZeRO

**Thesis:** ZeRO is DDP that stops storing the same thing N times, and pays for it in network traffic.

Beats:

1. DDP: replicate everything, all-reduce grads (2x params on the wire). Overlap: each layer's grads go out while the previous layer computes its backward. Exposed comms when comms > compute.
2. ZeRO through the backpacking metaphor: one person carries the tent, one the stove, one the axe, and they share at night.
3. Worked toy: 3 layers × 3 params on 3 GPUs. GPU0 holds a0, b0, c0, gathers a1, a2 just in time for La, then drops them.
4. Stages: 1 shards optimizer states, 2 adds grads, 3 adds params. Comms: ZeRO-1/2 ≈ 2x params (same as DDP), ZeRO-3 ≈ 3x (2 all-gathers + 1 reduce-scatter).
5. Will the network keep up? `comms_time = mult × bytes × params_B / GBps` vs `compute_time`. IDEFICS-80B: comms 11s, compute 18s, measured 49s; 90 TFLOPS on ZeRO-3 vs 150+ on Megatron TP+PP+DP. A 5x slower network → 56s of comms.
6. DeepSpeed 176B on V100: 100Gbps IB < 20 TFLOPS/GPU, 800Gbps > 40.
7. GAS multiplies ZeRO-2/3 comms, not ZeRO-1.
8. Scale problems: 1024 GPUs × MBS 32 = 32k GBS; slow inter-node used for everything. ZeRO++ hpZ / FSDP `HYBRID_SHARD`: shard inside a node, replicate across nodes.

Visuals: the memory bar from #5 sliced across GPUs per stage; the backpackers; timeline lanes showing overlap and exposed comms.

Sources: `training/model-parallelism/README.md` (Data Parallelism, ZeRO, ZeRO with multiple replicas, Inter-node speed requirements to use ZeRO), `network/README.md` (Single node training, Comms and compute overlap), `training/performance/README.md` (Gradient Accumulation).

Gaps: the per-stage memory formulas from the ZeRO paper (the repo only shows its diagram).

---

## 8. Cutting the model

**Thesis:** when one GPU can't hold the model, you can cut it by layers, by matrices, by sequence, or by experts, and each cut has its own communication bill.

Beats:

1. Naive vertical split: layers 0-3 on GPU0, 4-7 on GPU1. Only one GPU works at a time.
2. Pipeline parallelism: split the batch into micro-batches so stages overlap. The bubble. `chunks` = GAS; GBS = mbs × chunks × dp (8 × 32 × 4 = 1024). Schedules: GPipe → interleaved 1F1B → looped depth-first → breadth-first → DualPipe. Practical pain: batch-first tensors at stage boundaries, no conditional control flow, a heavy embedding stage.
3. Tensor parallelism (Megatron): `Y = GeLU(XA)`, split A by columns → each GPU applies GeLU independently → split the next matrix by rows → one reduction at the end. Attention heads are already independent. Must stay inside the fast domain (a node, or an NVL72). Async TP overlaps the all-gather with partial matmuls.
4. Sequence parallelism, why: attention compute is quadratic; at long sequences even bs=1 doesn't fit.
   - Ulysses: shard tokens, all-to-all to switch to head sharding for attention, all-to-all back. Example: 8K tokens, 128 heads, 8 GPUs → 1K tokens and 16 heads each.
   - Ring attention: queries stay, keys/values travel around the ring.
   - Megatron SP: pairs with TP for norm/dropout. Comms `4Nh` vs Ulysses `4Nh/P` per layer.
5. Expert parallelism: each expert on its own GPU; tokens move instead of weights (all-to-all).

Visuals: the GPipe bubble chart morphing through schedules; a matrix splitting by columns then rows; a sequence strip reshuffling through all-to-all; KV blocks circling a ring.

Sources: `training/model-parallelism/README.md` (Pipeline Parallelism methods, Tensor Parallelism, Sequence Parallelism, TP+SP, Expert Parallelism).

Gaps: the bubble fraction formula, TP's forward/backward conjugate ops, and nearly all of EP (the repo only links two references).

---

## 9. 3D parallelism: which strategy when

**Thesis:** real runs combine the cuts from #8, and the hardware topology decides which cut goes where.

Beats:

1. DP+PP: DP sees 2 GPUs; each one secretly enlists a PP partner.
2. DP+PP+TP: the 3D cube (8 GPUs minimum). DP = N_GPUs / (TP × PP).
3. ZeRO with PP: only stage 1. Stage 2 would need a reduce-scatter per micro-batch, and PP already cuts grads by 1/PP.
4. Map to topology: TP inside the scale-up fabric, PP across nodes (point-to-point only), DP outermost. Collectives per strategy (DDP 2x, ZeRO-3 3x, TP 2 all-gather + 2 reduce-scatter, PP send/recv).
5. The decision tree: single GPU / single node / multi-node; fits vs doesn't; fast vs slow inter-node.
6. Inference flips it: TP lowers latency, PP raises throughput. Llama 405B: TP=8 talks to 7 peers, PP=8 to 2; TP=4 + PP=4 can win.
7. Coda: FlexFlow searches the space automatically (sample, operator, attribute, parameter).

Visuals: a 3D grid of GPUs colored per axis, overlaid on nodes and racks; the decision flowchart.

Sources: `training/model-parallelism/README.md` (DP+PP, DP+PP+TP, ZeRO DP+PP+TP, Parallelism network collectives, Which Strategy To Use When, FlexFlow), `inference/README.md` (Model parallelism), `training/performance/README.md` (Batch sizes).

---

## 10. The network

**Thesis:** inter-node bandwidth decides whether many GPUs act like one big GPU, but the numbers on the spec sheet mislead in both directions.

Beats:

1. Three networks per node: frontend, backend (the one that matters), out-of-band. Bits vs bytes; unidirectional vs duplex (measuring 235 of "600" is really 80% of 300).
2. Intra-node: NVLink generations, NVSwitch, PCIe an order of magnitude behind. AMD MI300X: 64GBps peer-to-peer vs 448 all-to-all, so 2 GPUs run 6.5x slower than 8.
3. The 2B DDP story: 1 GPU (no comms) → 8 GPUs (16GB of grads / 300GBps = 0.053s vs 0.42s compute, fine) → 4 nodes at 200Gbps (0.64s > 0.42s, comms-bound). Then do it for 20B and 200B.
4. Real throughput: ~80% of spec, and only at large payloads (all-reduce busbw 1.3GBps at 32KiB → 235GBps at 16GiB). 1 × 4GB is ~3x faster than 1000 × 4MB → bucket your grads.
5. The surprise: P6-B200 links are 18x apart (900 vs 50GBps), but a 4GiB all-reduce across 4 nodes is only ~2x slower than in one node. NCCL reduces inside the node first, so only 9.7% of the bytes leave, and all 8 NICs work at once.
6. Converting busbw to a per-accelerator wire rate: `busbw × (k-1)/(n-1)` → 36.5GBps, 73% of spec.
7. Latency and hops; node proximity (placement groups); shared networks make tuning impossible (JeanZay before BLOOM).

Visuals: a two-tier topology (NVLink inside, NICs out); timeline lanes; the payload S-curve; a hierarchical all-reduce animation where only a thin slice crosses between nodes.

Sources: `network/README.md` (Introduction, Cluster networks, Intra-node networking, Understanding why inter-node network speed is of a huge importance, Important nuances), `insights/ai-battlefield.md` (Network).

---

## 11. Inference: prefill vs decode

**Thesis:** generation is two different workloads glued together: one is compute-bound, the other is starved for memory bandwidth.

Beats:

1. Prefill processes the prompt in parallel (compute-bound, sets TTFT). Decode makes one token at a time (memory-bound, sets TPOT).
2. Memory: weights (2 bytes bf16 … 0.53 bytes MXFP4) + KV cache + activations.
3. KV cache formula: `bytes × 2 × layers × hidden × kv_heads / heads` per token. Llama-3.1-8B: 0.131MB/token; 128 sequences × 1024 tokens = 17.2GB. GQA is 4x smaller than MHA here, MQA 32x; MLA compresses further.
4. Why decode is memory-bound: linear layers become compute-bound as batch grows; attention over the KV cache stays memory-bound at any batch size.
5. Batching: static (everyone waits for the longest) vs continuous (swap finished sequences out each step). Paged attention treats KV memory like OS pages.
6. Speculative decoding: the "another lemon tree" walkthrough. Same compute or more, lower latency. Best on input-grounded tasks and greedy decoding; ngram prompt lookup needs no draft model.
7. Guided generation, and the trick where a schema's fixed keys are prefilled instead of decoded.
8. Metrics: TTFT, TPOT from reading speed (250-700 WPM; prose ~1.2 tokens/word, code ~3.3). Collapse everything to prefill throughput and decode throughput. Percentiles.

Visuals: the roofline idea from #1 with prefill on the right and decode on the left, batch size sliding it; the KV cache growing per token; MHA/GQA/MQA heads sharing K/V; continuous batching lanes.

Sources: `inference/README.md` (Concepts, Key inference performance metrics, Benchmarks).

Gaps: paged attention and continuous batching are one paragraph each; the vLLM / Orca papers would be needed for depth.

---

## 12. Reading loss curves

**Thesis:** a loss curve is a heartbeat; most spikes have recognizable shapes, and some aren't real.

Beats:

1. Gallery: the failed pre-BLOOM 104B (fixed for BLOOM-176B by bf16, cleaner data, embedding layernorm), the near-perfect BLOOM-176B (one spike, recovered in 200 steps), a grokking moment (4 → 2.5 in 480 samples).
2. Spike taxonomy: fast recovering, slow recovering, not fully recovering, non-spike divergence, per-dataset spikes in mixed training.
3. Causes and fixes:
   - bad data pockets; the problem builds many steps before the spike
   - PaLM: roll back and skip the batches
   - init std: Megatron's 0.02 was too big; `sqrt(1/(3·H))` = 0.00482 for H=14336
   - scaling Q and K by `sqrt(norm)` before the matmul instead of the product after (`n(A·B) = (√n A)(√n B)`)
   - Adam epsilon and time-domain correlation
4. Spikes that aren't spikes: resume artifacts. DataSampler image/text ratio drift; repeated data after a PTL resume reporting a falsely low loss that "jumps" once new data arrives.

Visuals: the repo's real loss plots, annotated; init std vs hidden size; Q·K magnitudes blowing past fp16 range.

Sources: `training/instabilities/README.md`, `training/instabilities/training-loss-patterns.md`, `training/dtype.md` (Changing precision post training).

---

## 13. Things will break

**Thesis:** at scale, hardware failure is a schedule, not an accident, so design for it.

Beats:

1. One failing GPU out of 8 takes down the whole node; new accelerators can fail at up to 10% early on.
2. Keep 5-10% spare nodes; prefer fixed allocations (bad GPUs get weeded out) over dynamic ones (you inherit other users' rejects).
3. Checkpoint cadence: BLOOM saved 2.3TB in 40s every 3h → 720 saves → 8h total = 0.37% of training. With 5x slower IO, 2%. Keep the last 2 checkpoints local.
4. Multi-replica recovery: copy state from a healthy replica, lose one iteration instead of hours (torchft; PyTorch 2.14 in-place process group rebuild).
5. Operational switches: job arrays, kill switch, save switch, watchdogs (disk space, hangs), preemption signals.

Visuals: a cost curve for checkpoint interval (lost work vs save overhead); a replica healing from its twin.

Sources: `training/fault-tolerance/README.md`.

Gaps: the repo has no failure-probability-vs-cluster-size math and no optimal checkpoint interval formula (e.g. Young/Daly). This is the most operational video of the set.
