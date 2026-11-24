# Real-weight convolution qualification

The resident path now evaluates three isolated layers from the pinned FrameTop
v20a model on four real recorded infrared crops. All **2,207,744 output values**
pass the frozen numerical rules, including held-out crops. The final calibration
and held-out runs pass **3,558 restoration audits**, with one HMX owner, the
existing 40 ms timeout, and zero service restarts.

This is layer-local numerical qualification. Inputs to later layers are ncnn's
intermediates; DSP errors are not propagated through a complete landmark graph.
Full graph execution, landmark/confidence accuracy, broader recordings and
sustained tracking remain separate milestones. The prototype uses one bounded
spatial tile per request and is not a real-time convolution implementation.

The [sanitized evidence](../evidence/convolution-validation.json) records model,
source, crop/tensor and executable hashes, reference build options, tolerance
derivation, every case's error/timing summaries, session counts and raw-report
digests. Raw weights, images, crops and tensor files are excluded from releases.

## Fixed references and data

| Component | Pin |
|---|---|
| FrameTop integration | `edd7e61fdaa76bfa71f532cf8cb8e0a6bf60d729` |
| v20a model repository | `03d7695fecb80f6cf8f3aa9321de130b5efd08b3` |
| ncnn | Release `20260526`, commit `e54f7b1f88434e1d844ea0551b880a1cfb079ce1` |
| ncnn full-source ZIP SHA-256 | `754659d6fe65545cf2ef4483ffb84526fea631f8764c44b150f1601d0fb4004b` |
| Param SHA-256 | `54e08f284ff6f121ff00f4a9c65afc38c0dbb22a32f4af44fc4e0fde6c4a1bbd` |
| Weight SHA-256 | `b7c3d1be232725a5a143e2648858ec5aec9596c7b83c6a3c508479b04057e00c` |

The 2,046,732-byte weight file matches
[FrameTop's pinned catalog](https://github.com/Frametop/frametop/blob/edd7e61fdaa76bfa71f532cf8cb8e0a6bf60d729/hands/models.json).
The ncnn release/commit/archive hash come from its
[Makefile](https://github.com/Frametop/frametop/blob/edd7e61fdaa76bfa71f532cf8cb8e0a6bf60d729/hands/Makefile).
[ModelBin](https://github.com/Tencent/ncnn/blob/e54f7b1f88434e1d844ea0551b880a1cfb079ce1/src/modelbin.cpp)
and [Convolution](https://github.com/Tencent/ncnn/blob/e54f7b1f88434e1d844ea0551b880a1cfb079ce1/src/layer/convolution.cpp)
define the pinned tags, weight order, bias, padding and fused activation semantics.
The weights are attributed to **Frametop (DeeJanuz)** under CC BY-NC 4.0;
they are external data, not part of FrameNPU's MIT distribution.

The private recording supplies Beginning/EndSnapshot grayscale SLAM-camera PNGs.
Four explicit hand/controller ROIs use the affine and replicated grayscale
normalization from pinned
[nets.cpp](https://github.com/Frametop/frametop/blob/edd7e61fdaa76bfa71f532cf8cb8e0a6bf60d729/hands/track/nets.cpp):
warp bilinear to 224×224, border zero, contrast **none**, and three identical
FP32 channels multiplied by `1/255.f`. ROIs are manually selected, not produced
by a palm detector. Source-image hashes, centers/sizes/rotations and crop hashes
are pinned in the evidence. Two crops calibrate the limits and two are held out.
They share one recording/scene and do not represent a broad accuracy dataset.
Held-out means excluded from tolerance calibration; model-training exclusion
has not been established.

The strict CPU oracle uses one thread, unpacked FP32, no FP16/BF16/Vulkan,
and disabled Winograd/SGEMM. It is built with `-O2 -ffp-contract=off
-fno-fast-math`, avoiding ncnn's Release-mode fast-math defaults. These oracle
options intentionally differ from FrameTop's ARM FP16 execution settings.
The actual compiler, architecture, archive/library/tool hashes and options are
recorded. Preparation used Python 3.14.9, NumPy 2.5.5 and Pillow 13.0.0; those
optional tools are not runtime dependencies of the host library.

## Preflight and selected layers

[convolution.py](../src/framenpu_cdsp/research/convolution.py) validates the complete pinned
77-layer / 90-blob DAG before device access: producer order, unique blobs/names,
supported attributes, shapes, group/weight counts, tagged FP16/FP32 storage,
four-byte alignment, FP32 biases, finite values and exact binary EOF. It supports
the explicit operator subset needed for this graph and rejects other semantics.
The existing `inspect-ncnn` CLI remains an inventory command; this is a separate
strict preflight API, not an arbitrary ncnn graph executor.

| Layer | ncnn blobs | Input CHW | Output CHW | GEMM reduction |
|---|---|---|---|---:|
| `convclip_0` | `in0` → `2` | 3×224×224 | 24×112×112 | 27 |
| `conv_10` | `3` → `4` | 24×112×112 | 16×112×112 | 24 |
| `conv_12` | `8` → `9` | 64×56×56 | 16×56×56 | 64 |

The stem is 3×3, stride 2, dilation 1, padding top/left 0 and bottom/right 1,
with FP32 bias and clip [0,6]. The two pointwise layers are 1×1, stride 1,
unpadded, with FP32 bias and no activation. K=64 exercises two distinct depth
tiles within one complete HMX transaction.

The ncnn exporter records each target's input and output from the actual graph
prefix. A companion ncnn network runs the exact isolated layer segment after
rounding its input to FP16 and promoting back to FP32. A separate FP64 patch
lowering implementation checks both oracles within `1e-4`. This separates input
quantization error from backend arithmetic and detects weight-order/padding
mistakes independently of the DSP implementation.

## DSP path

`ResidentConvolution` lazily opens its session after input shape/range validation.
It transposes ncnn `[output,input,kh,kw]` weights into GEMM `[K,N]`, pads K to
32/64 and N to 32, then uploads/prepacks weights once per layer session. Both
crops of a split reuse that same session and immutable FP32 biases. It lowers
only one 32-row patch tile at a time in channel/kernel order; no full im2col
tensor is materialized. Spatial/channel tails are checked and excluded from
the logical output. Unused spatial rows produce the padded bias/clip result.

[resident_conv.c](../tests/resident_conv.c) is a 3,796-byte tag-2 derivative
of the [resident GEMM proof](RESIDENT_GEMM.md). It uses the same 32 KiB partition,
nonce/content/sequence/bounds checks and cached entry cleanup. Biases occupy
128 bytes at resident control offset +256, outside the cached data window.
The content identity includes weights, biases and activation mode; immutable
bias bytes are checked before submission and after loading/return. Descriptor
word 25 selects none/clip and is bound to native session state.

Worker 0 performs one clear/load/store transaction over distinct depth tiles.
The final FP16 HMX result is promoted to FP32; native scalar code adds FP32 bias
and applies clip. It counts nonfinite accumulators before activation so clip
cannot hide overflow. Only the existing backed-up/restored 8 KiB VTCM is used,
including a 128-byte bias staging vector. The existing executable slot, wire
ABI and per-request power/40 ms timeout policies are preserved.

Every output is checked before host reclamation or power-down. Last-tile RMS
checks also run inside completion validation. Unexpected numeric/metadata/cache
state retains resources under RecoveryRequired. Normal close restores all owned
32 KiB and borrowed spans, removes the dispatch patch, then retires ownership.
Three guarded native trap controls pass dirty-output cleanup and invalidation;
they do not establish arbitrary corruption or pipeline-stall recovery.

## Frozen numerical rules

The arithmetic rule is checked for every value against the FP16-input ncnn
companion: `abs(actual-qref) <= atol + rtol*abs(qref)`. All outputs and raw
accumulators must be finite. `rtol=2^-10` is a one-ULP FP16 allowance;
`atol=rtol*max_abs_bias + 1e-4` accounts for cancellation around bias and FP32
reduction differences. Neither term is fitted to observed DSP errors.

Separate maximum/RMS limits gate total error against the original FP32-input
oracle. They are 1.25 times the worst **calibration-only CPU precision estimate**:
the sum of input-quantization and ideal final-FP16-store errors. Maximums round
up to `1e-4`, RMS to `1e-5`. Held-out errors cannot widen those limits.

| Layer | Arithmetic atol | Arithmetic rtol | Full FP32 max-absolute limit | Full FP32 RMS limit |
|---|---:|---:|---:|---:|
| Stem | 0.0038562894 | 0.0009765625 | 0.0317 | 0.00151 |
| Pointwise | 0.0023896649 | 0.0009765625 | 0.0077 | 0.00062 |
| K=64 | 0.0048396780 | 0.0009765625 | 0.0081 | 0.00119 |

The final tolerance-file SHA-256 is
`3dd2129fd21a30dcdb4ca60f87e27a9b206aa7aefd0e00c9cad8ecd604e44ded`.
Both final calibration and held-out runs use that file. An earlier calibration
checked only the arithmetic rule; it is excluded from final counts. No kernel
arithmetic optimization or tolerance widening followed held-out results.
These are provisional layer-regression limits for this corpus, not landmark
accuracy acceptance criteria.
Final host review also makes every checker exception retain resources. Repeating
the complete matrix after that change produces identical output hashes for all
2,207,744 values; the evidence records the earlier report digests separately.

## Results and timing

Worst held-out errors against the original FP32 oracle are:

| Layer | Maximum absolute error | Worst crop RMS | Maximum error against FP16-input ncnn |
|---|---:|---:|---:|
| Stem | 0.0140700 | 0.000785676 | 0.00194645 |
| Pointwise | 0.00543642 | 0.000328457 | 0.00387049 |
| K=64 | 0.00533867 | 0.000662564 | 0.00388694 |

Across all four crops the stem's maximum is 0.0228217 and pooled RMS 0.000817985.
Input FP16 quantization dominates that layer's error. All mixed-rule ratios
remain below 0.48. The stem has 1,204,224 checked values, pointwise 802,816 and
K=64 200,704. Right/bottom padding, channel tails and clip boundaries are included.

The final runs perform 3,528 convolution tile jobs in six layer sessions. Each
session loads weights/biases once. Including reconciliation/load/close and
dispatcher helpers, all 3,558 audits pass: 3,546 HMX commands and 12 helpers.
Original gate/callback, firmware prefix and power hashes match after each split;
the service keeps its PID with zero restarts. The three-fault pilot passes
24 separate audits. Earlier resident code replacement/reuse passes 63 audits
after the extension, and ordinary GEMM/activation/vector/XOR checks pass 78 more.
These supplemental counts are separate from the final convolution matrix.

| Held-out layer body | Time per crop | Host patch lowering included | Median command |
|---|---:|---:|---:|
| Stem, 392 requests | 6.54–6.60 s | 129–134 ms | 15.1–15.2 ms |
| Pointwise, 392 requests | 6.34–6.54 s | 158–171 ms | 14.7–15.2 ms |
| K=64, 98 requests | 1.72–1.74 s | 109–113 ms | 14.9–15.5 ms |

Body timing includes patch lowering, input encoding, command execution,
numerical/restoration audits and result assembly; it excludes full-input
preflight and session open/close, which are reported separately. The DSP wrapper
median is about 5.54 ms: code publication ~0.72 ms, output conversion plus scalar
bias/clip ~1.15 ms and full-window cache cleanup ~3.59 ms dominate the short HMX
intervals. Maximum wrapper is 5.644 ms and publication-to-completion 7.260 ms.
Counters have 52.08 ns resolution and no overhead subtraction. Host/DSP phase
percentiles are not additive; resident upload is nested within prepare timing.

This is audited correctness timing. The next performance work can batch bounded
spatial tiles and reduce publication/cache/power overhead while preserving the
frozen rules and existing completion limits. Full-graph error propagation and a
larger independent corpus are needed before tracking accuracy claims.

## Reproduction

Keep source weights, recordings and generated artifacts in a private directory.
Acquire the param/bin from the pinned
[v20a revision](https://huggingface.co/DeeJanuz/frametop-hands-models/tree/03d7695fecb80f6cf8f3aa9321de130b5efd08b3/v20a)
and verify the hashes above. The crop recipe in the preparation tool applies
to the recorded snapshots identified by the evidence; other recordings need
their own explicit ROIs, corpus pin and calibration.

~~~sh
python3 -m pip install numpy==2.5.5 Pillow==13.0.0
python3 tools/build_ncnn_reference.py --work-dir /path/private/reference
python3 tools/prepare_conv_reference.py --recording /path/to/recording.dataset \
    --model /path/private/model --reference /path/private/reference/ncnn_reference \
    --layer-reference /path/private/reference/ncnn_layer_reference --output /path/private/corpus
python3 tools/analyze_conv_reference.py --model /path/private/model --corpus /path/private/corpus
python3 handoff-tools/build_mw.py tests/entry_resident.S tests/resident_conv.c \
    /path/private/resident_conv.fjob /path/private/native-build
~~~

After inspecting supported-device idle/power state, run a small control and
guarded pilot, then complete calibration and held-out validation. The held-out
runner requires a completed calibration report with identical policy/payload:

~~~sh
sudo python3 tests/hardware_convolution.py --run-on-device --probe resident_conv.fjob \
    --model model --corpus corpus --layers convclip_0 --tiles 1 --output /tmp/conv-control.json
sudo python3 tests/hardware_convolution.py --run-on-device --probe resident_conv.fjob \
    --model model --corpus corpus --tiles 1 --fault-control --output /tmp/conv-pilot.json
sudo python3 tests/hardware_convolution.py --run-on-device --probe resident_conv.fjob \
    --model model --corpus corpus --split calibration --output /tmp/conv-calibration.json
sudo python3 tests/hardware_convolution.py --run-on-device --probe resident_conv.fjob \
    --model model --corpus corpus --split heldout --calibration-report /tmp/conv-calibration.json \
    --output /tmp/conv-heldout.json
~~~

Native jobs remain trusted supervisor code. Failures stop without automatic
gate clearing, power recovery or service restart.
