# Custom hand tracking on Frame: next milestones

Assessment: 2026-11-15, against FrameNPU 0.1.0 and the pinned FrameTop v20a
landmark graph described below. Bounded [two-depth-tile execution](MULTITILE.md)
now passes numerical, timing and restoration checks, as do
[HVX packing for full aligned tiles](HVX.md) and [HVX output conversion](READBACK.md).
The [cached-code investigation](HMX_CACHE.md) identifies uncached instruction
fetch as the earlier resident loop's bottleneck. The offline resident planner,
bounded resident GEMM and [three real-weight convolutions](CONVOLUTION.md) now
have separate validation. A [complete streamed landmark prototype](LANDMARK.md)
now includes the remaining operators and aliases, with host FP32 intermediates.
All 77 layers, 90 blobs and four outputs pass on four recorded crops under
frozen provisional limits. Its audited body still takes 71–79 seconds per crop.
Resident intermediates, weight reuse, broad accuracy and sustained frame rate
remain unverified.
An opt-in [direct dispatcher](DISPATCH.md) now passes CPU/DSP
and HMX device tests with a worker join and reversible installation. Median
warm empty submission fell from 4.29 ms to 0.73 ms; complete 32x32 HMX GEMM
submission fell from 29.34 ms to 25.36 ms. Tracking coexistence checks remain
pending. The cost examples below describe the default benchmark backend
and retain the earlier roughly 13 ms research estimate; it is not a fixed cost
across device conditions.

## What already works

The [device baseline](VALIDATION.md) establishes custom C jobs, padded HMX
matrix products, DSP accumulation/activations and a serialized XOR model on
SteamOS 0.5.8 / CDSP ABI 9 / Hexagon v75. Firmware identity and completion checks
are implemented. These are the foundations for a backend; the current model
format executes only sequential Dense, ReLU and Clip layers.

The default benchmark path has three costs that dominate a convolutional workload:

- Every submission invokes the stock FLOPS benchmark, costing roughly 13 ms.
  [Device.matmul](../src/framenpu_cdsp/device.py) submits each depth/output tile
  separately, reads partial results into Python and submits DSP additions.
- Each job uploads code and operands, backs up/restores borrowed memory, and
  scopes HMX power to that submission. Weights and intermediate tensors have
  no persistent ownership.
- The stock dispatcher releases its completion/ownership window after roughly
  50 ms. The existing host limit is 40 ms; increasing it does not extend
  ownership. See the [failure contract](ABI.md#completion-window-and-failures).

`JobResult.elapsed_ms` starts after upload and HMX power setup and ends before
power teardown, restoration and acknowledgment. Measure around the entire
inference call when comparing end-to-end latency.

## Workload and capacity audit

This audit uses the locally retained graph from model revision
`03d7695fecb80f6cf8f3aa9321de130b5efd08b3`, with FrameTop source revision
`edd7e61fdaa76bfa71f532cf8cb8e0a6bf60d729`. It does not assume future model
revisions have the same topology. The landmark param SHA256 is
`54e08f284ff6f121ff00f4a9c65afc38c0dbb22a32f4af44fc4e0fde6c4a1bbd`.
The [model graph](https://huggingface.co/DeeJanuz/frametop-hands-models/blob/03d7695fecb80f6cf8f3aa9321de130b5efd08b3/v20a/hand.ncnn.param)
has 77 layers / 90 blobs: 31 Convolution, 16 ConvolutionDepthWise,
4 InnerProduct, 10 BinaryOp, 2 Pooling, 11 Split, Input, Reshape and Squeeze.
Fused activations include clip and sigmoid.

For ordinary convolution, MACs are
`output_height * output_width * weight_count`; depthwise uses the same formula
with its grouped weight count. Dense MACs equal weight count. Applying these
formulas and the graph's explicit strides/padding gives:

| Computation | MACs per crop | Share | Initial implementation |
|---|---:|---:|---|
| Thirty 1x1 convolutions | 115,781,120 | 79.52% | HMX with prepacked weights |
| Sixteen depthwise convolutions | 21,610,176 | 14.84% | HVX 3x3 / 5x5 kernels |
| Initial 3x3 convolution | 8,128,512 | 5.58% | Tiled input packing + HMX |
| Four dense heads | 86,016 | 0.06% | Compare HVX and HMX for small shapes |
| Total | 145,605,824 | 100% | Excludes preprocessing and non-MAC operators |

The first convolution alone is GEMM `(M,K,N) = (12544,27,24)` after lowering
the 224x224x3 input to patches; its output is 112x112x24. With the current
32x32x32 tile schedule this requires `ceil(12544/32) = 392` HMX submissions.
At roughly 13 ms of stock benchmark per submission, that is approximately
5.10 seconds of benchmark overhead alone. This is a hypothetical mapping,
not a measured convolution; the current public matrix API also caps each
dimension at 256.

Memory imposes an independent limit:

| Item | Logical storage before packing/alignment |
|---|---:|
| 224x224x3 input, FP16 | 294 KiB |
| Largest activation, 112x112x64 FP16 | 1.53125 MiB |
| 1,000,160 FP16 weights + 11,552 FP32 biases | 2,046,528 bytes, about 1.95 MiB |
| Current combined input/output job arena | 128 KiB |
| Current HMX scratch span used with save/restore | 8 KiB |

The largest activation is not the peak live tensor set: residual inputs and
operator outputs can coexist. Later layers also have 672 channels, exceeding
the current matrix dimension limit. Tiling must live inside the executor.

Research identified 8 MiB total VTCM and a separate gaze reservation. This is
not a free-capacity measurement. Neither the remaining VTCM nor the 32 MiB
shared DDR mapping can be treated as available model storage. The existing
job spans are borrowed under the operation gate; they are not allocations
that can survive its release.

At two hands, two views and 30 Hz, landmarks require
`145605824 * 4 * 30 = 17.4727 GMAC/s`. Four crops have a total 33.33 ms frame
budget, including transfers, preprocessing and tracking. Equal serial crops
would have less than 8.33 ms each after reserving time for that other work.
These are workload targets, not demonstrated throughput.

## 1. Establish dispatcher ownership and measure overhead

Start with a minimal direct job path that bypasses the FLOPS benchmark and
cooperates with existing firmware clients. It needs:

- A mapped dispatch/wakeup/return path, with explicit ownership of workers,
  executable storage, shared DDR, VTCM and HMX power. Identify a valid extension
  point before assigning an operation selector or choosing memory addresses.
- A versioned command/completion protocol with sequence and session generation,
  descriptor bounds, and explicit acknowledgment. A late completion must not
  be mistaken for a new job after reconnect/reset.
- Bounded work segments and cancellation at a completed tile boundary. Host
  timeout or disconnect cannot authorize reclaiming memory or powering off an
  engine that may still be running. All participating workers must return
  before releasing their resources.
- Required cache maintenance and publication barriers for code, descriptors
  and tensors; an uncached host mapping alone does not prove every DSP access
  is coherent.

Instrument host claim/upload/power/notification/wait/download/cleanup and
DSP queue/packing/transfer/HMX/HVX execution separately. Compare an empty job,
a copy and a repeated resident GEMM at the same verified clock setting.
Report p50/p95/p99, not just the best iteration. Do not infer performance
from advertised TOPS or extrapolate to unvalidated clock rates.

**Exit criterion:** repeated jobs bypass the benchmark, produce correct
results and restore idle state, while delayed acknowledgment, cancellation,
stale completions and another participating client cannot cause premature
reuse. The new path must establish its own completion contract before any
relaxation of the old path's 40 ms limit. Keep the existing backend for
regression comparisons.

## 2. Add resident tensors and a graph execution plan

Introduce tensor descriptors with shape, dtype, layout, strides, arena offset
and lifetime. Load/prepack weights once and keep intermediate activations on
the DSP. Python can remain the research control interface; the steady-state
loop should submit a crop or bounded graph segment, rather than individual
matrix tiles or layers.

Build a DAG plan: Split normally aliases storage, residual inputs stay live
until their last consumer, and shape operators require verified alias/layout
rules. Compute peak live bytes and allocator size including alignment,
packed weights, accumulators, packing buffers and double buffering. Make
DDR the backing store and use only explicitly owned VTCM as a tile cache.
Preserve gaze and FeatureNN reservations through the complete lifetime of
every allocation.

The 4 KiB executable reservation also needs a plan: either verify a larger
owned code region or load bounded kernel modules only after prior execution
has stopped, with the required instruction synchronization. Adding every
operator to the current runtime cannot assume unlimited code space.

**Exit criterion:** a planned residual block has a documented allocation map,
fits the verified reservations, reuses weights across runs and performs no
host round trips for intermediate tensors. No addresses are assigned solely
because a memory span currently contains zeros.

## 3. Validate real convolutions and numerical behavior

The [bounded convolution proof](CONVOLUTION.md) now pins the v20a weights,
strict ncnn CPU reference and four real recorded IR crops. Stem `convclip_0`,
pointwise `conv_10` and K=64 `conv_12` pass all 2,207,744 values under frozen
arithmetic and full-FP32-reference limits, including held-out crops. The final
runs pass 3,558 restoration audits without a service restart. This validates
isolated layers fed by reference intermediates; full-graph error propagation
and landmark/confidence tolerances remain to be established.

Pin the ncnn reference build, param/bin hashes and a deterministic crop corpus.
The [inventory command](../src/framenpu_cdsp/ncnn.py) remains separate from the
new [strict preflight](../src/framenpu_cdsp/research/convolution.py), which checks graph connectivity, inferred shapes,
attributes, binary lengths and supported semantics before device writes.
Decode tagged FP16/FP32 weights and FP32 biases, then transpose/prepack for
the chosen kernel layout. ncnn's
[binary reader](https://github.com/Tencent/ncnn/blob/master/src/modelbin.cpp)
and [convolution loader](https://github.com/Tencent/ncnn/blob/master/src/layer/convolution.cpp)
define tag/alignment, asymmetric padding and activation parameters; pin the
reference revision when implementing conversion.

Use `convclip_0` as the first real test: 3x3, stride 2, 3 input / 24 output
channels, left/top padding 0, right/bottom padding 1 and clip [0,6]. Compare
intermediate outputs against ncnn. Then validate a 1x1 layer and a layer with
reduction depth greater than 32 to exercise accumulation. Generate only the
input patches needed for each tile, rather than materializing a full im2col
tensor. Optimize the 1x1 path first because it accounts for about 79.5% of MACs.

The existing HMX path rounds each depth-32 product to FP16, then sums those
partials in FP32 on the DSP. Exact agreement with its tile-precision CPU
reference does not establish agreement with ncnn or hand tracking accuracy.
Measure real-weight error and saturation before choosing a production
accumulation scheme. Investigate retaining the HMX accumulator across depth
tiles, but validate its precision/range before relying on it. Record maximum
and RMS layer error and the effects on landmark positions and confidence.

**Exit criterion:** actual model weights and crops pass documented numerical
tolerances for the stem, pointwise and multi-depth cases, with separate kernel,
packing and end-to-end timings. Tolerances must be chosen from reference and
tracking accuracy results; the XOR test cannot supply them.

## 4. Complete one landmark graph

Current correctness result: [the streamed graph](LANDMARK.md) passes every layer
and all four outputs across two calibration and two held-out crops, with
20,351,360 checked values and 25,455 restoration audits. HVX depthwise/pooling/
residuals, native sigmoid and 26 storage aliases are implemented. The subsequent
[resident executor](RESIDENT_LANDMARK.md) now keeps weights and FP32 intermediates
in the verified DDR arena. Four crops pass all layers and heads, with two
changing inputs sharing one weight upload per session and full-arena teardown.
This closes the resident-intermediate correctness gap. Its uncached baseline
takes about 326 seconds per audited crop. The [gather optimization](DEPTHWISE_GATHER.md)
reduces that to 184–187 seconds with the same head bits; performance remains far
from integration.
The subsequent [resident cache entry](RESIDENT_CACHE.md) adds opt-in cached
instructions with checked normal/fault cleanup. Data stays uncached and all
existing resource, ownership and timeout limits remain in force.
[Linked operators](RESIDENT_LINK.md) now run from one image in owned DDR with
cross-layer requests and VTCM working copies: 52 requests and 1.66–1.69
seconds per frame at a 20 ms budget, .99 seconds of it native; opt-in
session-held HMX power brings the frame to 1.36–1.39 seconds. Native time
(depthwise and 1×1 convolution, about .45 seconds each) is again the largest
cost.

Add direct HVX depthwise kernels for 3x3 and 5x5, including stride 2 and
asymmetric padding. Add max/global-average pooling, residual add, fused clip,
sigmoid and validated Split/Reshape/Squeeze handling. Use a stable layout
across operators and fuse bias/clip/add where lifetimes and numerical
semantics permit. HVX should handle packing/conversion and vector operations;
HMX should handle suitable dense matrix work. Benchmark small tails/heads
before choosing their engine.

Validate each layer before comparing all four outputs for a complete crop:
`out0` screen landmarks, `out1` presence, `out2` handedness and `out3` world
landmarks. These names and the replicated grayscale RGB input normalized to
[0,1] come from the pinned
[FrameTop integration](https://github.com/Frametop/frametop/blob/edd7e61fdaa76bfa71f532cf8cb8e0a6bf60d729/hands/track/nets.cpp).
Preserve its crop transform, contrast settings and output interpretation.

**Exit criterion:** one full crop produces all four validated outputs with
resident intermediates. Then benchmark four crops sharing weights; batch them
only when measurements show a throughput benefit within the latency budget.
Initial support can target this pinned DAG while keeping custom weights
replaceable. General ONNX import and INT8 quantization follow a correct FP16
baseline and require their own operator/accuracy validation.

## 5. Integrate and qualify sustained tracking

Expose the executor through a native interface usable from FrameTop's C++
tracking loop. Initially retain palm detection, ROI construction and skeleton
reconstruction on CPU and replace landmark inference. Provide explicit backend
selection and CPU fallback for clean preflight/availability failures. A
poisoned DSP session must retain the existing recovery contract.

Measure one and four crops, queue age and dropped frames, full tracking
p50/p95/p99 latency, tracking accuracy/jitter, gaze timing/missed updates,
CPU time, power and thermals. Compare identical recorded inputs and a worn
headset baseline with and without sustained custom inference. A separate gaze
thread and a live service process do not establish unaffected gaze quality or
available compute bandwidth.

**Exit criterion:** sustained four-crop tracking meets the selected frame
deadline and accuracy target without unacceptable gaze regression or thermal
throttling. Only then assess palm offload and the additional reacquisition
workload. Real-time acceptance requires measurements on the intended firmware
and power setting.

## Next implementation deliverable

The persistent DDR region is now mapped and canary-checked; see
[MEMMAP.md](MEMMAP.md). The [HMX accumulator test](HMX_ACCUMULATOR.md) shows that
the recovered primitives share state across workers: full serialized
transactions pass, while deferred stores and concurrent transactions lose
isolation. Use one HMX owner for complete clear/load/store transactions, with
HVX workers for packing, depthwise and elementwise work. The earlier unchecked
aggregate issue estimates are not independent correct-GEMM throughput.

The [correct single-owner baseline](HMX_BASELINE.md) measures a resident load
slope of 7.41–7.44 GMAC/s, below the approximately 13.9 GMAC/s pointwise target.
The subsequent [cached-code comparison](HMX_CACHE.md) resolves this synthetic
loop bottleneck: 800 measured jobs and 32 warmups pass exactly at the same clock
configuration, with a fitted 1,264–1,269 GMAC/s resident slope. This establishes
that the old loop was limited by instruction fetch; it does not qualify a
distinct-weight model. A separate [two-block-load comparison](HMX_BATCHING.md)
passes another 800 measured jobs and 32 warmups, with about 2.52x improvement
in its uncached diagnostic. Preserve the distinction between the two results.

The [resident design and offline planner](RESIDENT.md) now define the memory,
session and cache contract, and produce a checked residual-block allocation map.
The planner does not load or execute a model. The separate
[resident GEMM proof](RESIDENT_GEMM.md) now uploads weights once and streams
distinct inputs through 32×K×32 products for K=32/64/128. It passes 300 measured
jobs plus warmup/reuse checks, 21 native rejections, three guarded traps and
code replacement/teardown, with 372 restoration audits. Ordinary SDK numerical
regressions pass another 78 audits, with no service restart. Its fixed layout
does not establish arbitrary graph support or model accuracy.

The measured resident GEMM commands take 12.4–12.6 ms median: per-request power
and conservative whole-window cache cleanup dominate. Amortize cached publication/cleanup
(roughly 1.5 ms per job) and fresh-session overhead (about 33 ms even for the
cached diagnostic), while preserving bounded ownership and cache publication.
The [public SDK unwind fix](FAULT_UNWIND.md) is now rebuilt and qualified with
nested guarded traps and numerical regressions. Distinct operand staging is now
qualified in the bounded resident proof. The subsequent real-weight convolution
proof also passes the stem/pointwise/deeper-reduction milestone against pinned
ncnn references and frozen limits.

The benchmark-free dispatcher, two-depth-tile GEMM jobs, full-tile HVX packing
and HVX output unpacking/conversion now have numerical and restoration validation.
FP32 accumulation, initialization and bias/ReLU remain scalar. Enforce the
allocation/ownership map before keeping weights across separate jobs. The offline
reference corpus and shape/binary preflight now exist. The current convolution
body still takes about 6.3–6.6 seconds for a 112×112 layer with 392 requests.
The complete streamed graph adds the remaining operators and passes propagated
output checks in 71–79 seconds per audited crop. The resident implementation
now passes the same four crops and produces bit-identical heads with weights
and FP32 intermediates in DDR. Its uncached depthwise gathers and many bounded
requests initially raised audited latency to about 326 seconds. Optimized gathers
now reduce that to 184–187 seconds: 2.38× faster depthwise native work and
54.5% fewer depthwise requests. [Cached resident instructions](RESIDENT_CACHE.md)
address instruction fetch next. Reduce the remaining uncached data access and
power/submission costs under the frozen rules, then benchmark more crops sharing weights.
Evaluate accuracy on a larger independent corpus before sustained integration.

Qualcomm's
[XNNPACK/HVX work](https://www.qualcomm.com/developer/blog/2026/03/bringing-xnnpack-hexagon-npu)
is a useful source of microkernel techniques. Its published kernel results do
not establish a compatible loader, scheduler or performance result for this
Frame firmware. The same runtime compatibility must be established separately
before treating a QNN/ONNX package as an alternative backend.
