# Complete landmark graph prototype

The experimental `StreamingLandmark` executor passes the pinned v20a
77-layer/90-blob landmark graph on four recorded infrared crops, checking every
layer and all four outputs under frozen provisional tolerances. The final
matrix checks 20,351,360 layer-output values and passes 25,455 restoration audits.
It is a streamed correctness proof with host intermediates; residency and useful
tracking performance remain open. Sanitized results and limitations are in
[landmark-validation.json](../evidence/landmark-validation.json).
The subsequent [resident executor](RESIDENT_LANDMARK.md) implements the same
pinned graph with weights and FP32 intermediates in DSP-accessible DDR. This
document and its measurements describe the original streamed implementation.

## Execution and ownership

This is a streamed correctness implementation. FP32 intermediate tensors live
on the host; bounded patches, weights and results cross the existing 128 KiB
borrowed job arena. It does not implement a resident graph or persistent weights.
There is no CPU arithmetic fallback. The CPU reference below is qualification
tooling, not part of execution.

| Graph work | DSP implementation |
|---|---|
| 31 convolutions and four dense heads | HMX, full reduction, padded M=N=32, K up to 704 |
| 16 depthwise 3×3/5×5 layers | HVX, separate FP32 multiply and add, bias and optional clip |
| Max pooling / global average | HVX max / ordered FP32 sum and reciprocal multiply |
| Ten residual adds | HVX FP32 add |
| Two sigmoid activations | Native range-reduced exponential polynomial and reciprocal |
| 11 Split layers, Reshape and Squeeze | Shared whole-tensor host storage; no copy/submission |

Convolutions quantize inputs to FP16 and store one full-depth HMX result in FP16
before FP32 bias/clip. They never store separate depth-32 partials. Output
channels and spatial rows are tiled independently. The 26 aliases extend root
lifetimes through their final consumer; logical peak live host tensor storage is
4,816,896 bytes. Reference arrays, Python object overhead and packed weights are
additional host memory. This number is not DSP residency or process RSS.

HVX uses v75 qf32 arithmetic with an FP32 conversion after each multiply/add.
Its rounding differs from the strict scalar CPU oracle and is covered by the
condition-scaled allowance below. Both zero-operand addition identities select
the original value exactly; sigmoid returns exactly 0.5 for zero. These preserve
the strict padded-row/channel checks without relaxing active-output tolerances.

Only SDK worker 0 uses HMX/HVX; context 4 joins and workers 1/2 return through the
existing entry. One direct Device owns a graph invocation and a host lock
serializes runs/close. Every command retains the 40 ms timeout and per-request
power/VTCM backup. Arithmetic modules use uncached arena data and code; they
introduce no cached aliases, enlarged VTCM allocation or firmware ABI changes.
Borrowed spans must restore before every acknowledgment.

Each module includes the current SDK fault unwind. The native ABI validates
opcode, exact lengths, alignment, flags, bounded tap/vector/depth counts and
reserved parameters. HMX clear/load/store remains one serialized transaction.
Vector nonfinite masks cover raw results before clip; the host checks them
before reclamation. Padded rows/channels and complete coordinate coverage are
also checked there. Any checker exception invalidates execution and retains
resources through `RecoveryRequired`; no automatic recovery is added.

Three separately built FJOBs fit the existing 4 KiB slot: HMX 2764 bytes,
HVX 2764 bytes, sigmoid 2564 bytes. The packaged runtime is unchanged. The generic
public `Device.matmul` API and its dimensions/arithmetic are also unchanged.

## References and provisional tolerances

Model/source/crop pins are unchanged from [convolution qualification](CONVOLUTION.md).
Weights and recorded data remain private; the model is external CC BY-NC 4.0.
The ncnn adapter's optional `all` export writes every named blob with light mode
disabled. Its original five-blob mode remains available. All five original
blobs, input and crop reproduce their step-3 hashes for every crop.

The strict CPU ncnn build remains the full-precision oracle. Independent CPU
formulas agree for all 76 non-input layers on four crops: 304 comparisons,
19,749,248 values. The per-value check is `1e-4 + 2e-5*abs(reference)`.
This verifies graph semantics independently of the HMX precision estimate.

An independent FP64 dot-product model estimates propagation of FP16 inputs and
one FP16 HMX store. Depthwise/residual/pooling use ordered FP32 CPU operations.
The two calibration crops alone fit per-layer maximum/RMS limits: twice the
estimated error plus .002 maximum/.0002 RMS reference scale, with absolute
floors .002/.0002. Head ceilings are fixed before device observation:

| Output | Values | Maximum / RMS ceiling |
|---|---:|---:|
| out0, screen landmarks | 63 | 1.0 / .25 crop-coordinate units |
| out1, presence | 1 | .005 / .005 |
| out2, handedness | 1 | .005 / .005 |
| out3, world landmarks | 63 | .003 / .001 model units |

Actual frozen caps may be tighter than these ceilings. Policy SHA256:
`381a033d6f4c10047998dfe42abb8cd6b18a756741c8a21690a3e6d6dc951f7b`.
Heldout crops cannot fit or widen limits. They come from the same scene;
heldout means excluded from tolerance fitting, not proven absent from training.
These are provisional regression limits, not tracking-accuracy requirements.

The hardware runner also checks every arithmetic chunk against an independent
CPU calculation from the *actual propagated DSP inputs*. HMX uses a mixed
`1e-4 + 2^-10*(abs(reference)+max_abs_bias)` allowance. HVX uses
`2e-5 + 2e-5*abs(reference)` plus a condition-scaled roundoff allowance:
`2^-22 * operation_count * sum_abs_terms`. Depthwise counts 2*taps+1,
average pooling counts 2*spatial+1; residual counts 1 and max needs no rounding
term. Sigmoid is checked within 2e-6 against its actual HMX logit. Every value
also passes the frozen propagated maximum and final-layer RMS limit against
ncnn. Aliases compare all output-blob hashes and check shared storage contents.

The completed heldout runner requires a completed calibration report with the
same native jobs, host/native source hashes, corpus manifest and tolerance pin.
Diagnostic crop/layer stops are explicitly incomplete and cannot qualify it.

## Hardware evidence

The final matrix covers two calibration and two held-out crops: 308 layer checks,
360 blob checks including aliases, and all 512 final head values. Independent
readback of the 16 saved head tensors reproduces their hashes, maximum errors
and RMS errors. Every arithmetic chunk is checked before resource reclamation;
complete coordinate coverage and each layer's RMS are also required there.

| Head | Held-out maximum error | Worst held-out crop RMS | Frozen maximum / RMS |
|---|---:|---:|---:|
| out0, screen | .108841 | .0424614 | 1.0 / .25 |
| out1, presence | .000117105 | .000117105 | .00200023 / .000200345 |
| out2, handedness | .000778735 | .000778735 | .002 / .00130716 |
| out3, world | .000195328 | .0000664248 | .002 / .0002 |

Across all four crops the screen maximum/worst-crop RMS are .397690/.184365, world maximum
is .000195328, and handedness maximum is .00117379. The largest local arithmetic
tolerance ratio is .8086; the largest propagated RMS ratio is .8980. No tolerance
was widened after device observation or heldout testing.

Each graph uses 6,356 arithmetic submissions: 3,383 HMX GEMMs, 2,916 depthwise,
25 max-pool, 2 global-average, 28 residual and 2 sigmoid jobs. The four graphs total
25,424 submissions. Seven normal controls, seven native rejections, seven
guarded traps and ten dispatcher helpers bring the main restoration count to
25,455. All arithmetic jobs carry the HMX flag for the existing power/VTCM
contract, including HVX/sigmoid work; 25,445 HMX-flagged audits do not mean that
many matrix products. Traps are intentionally injected before each operator's
body and verify entry unwind and the uncached sentinel; they do not establish
arbitrary pipeline-stall recovery.

The final controls include 22 distinct A and B depth blocks at K704, byte-exact
zero-product biases/zero-bias products, all-negative max pooling, signed
residual/average inputs, exact sigmoid zeros and saturation edges. The separate
final controls-only run passes 23 audits. An earlier `conv_16` pilot adds 4,133
audits, and the ordinary SDK GEMM/activation/vector/XOR regression adds 78. These
supplemental counts are excluded from the main matrix.

Three earlier validation stops exposed one-ULP zero-add identities and
`sigmoid(0)` rounding to .49999997. All workers returned normally; the validator
retained payload/power as designed. Read-only inspection identified the exact
errors, and explicitly authorized reboots reset state before further work.
These failed attempts and their reset history are retained separately in the
evidence. The final matrix uses the same final host/native sources across all
crops, keeps the service PID with zero restarts, and restores borrowed spans,
dispatcher, code prefix, gate/callback and original power after every session.

| Crop | Audited graph body | Host patch packing | Weight prepack | Executor close |
|---|---:|---:|---:|---:|
| Calibration controller | 76.86 s | 14.95 s | 123 ms | 4.93 ms |
| Calibration open | 78.32 s | 16.24 s | 128 ms | 4.93 ms |
| Heldout left controller | 77.18 s | 15.83 s | 90 ms | 4.93 ms |
| Heldout right controller | 71.45 s | 13.26 s | 200 ms | 4.92 ms |

Body timing includes per-layer CPU reference construction, patch packing,
submissions, numerical/restoration audits and assembly. It excludes constructor
weight prepack, whole-input preflight/copy and executor close. Reference work
totals 200–252 ms per crop. Host upload, power close, borrowed-span restoration
and acknowledgment have separate per-command phase records. The DSP wrapper
includes native operand copy/packing and arithmetic; these native subphases are
not separately instrumented in this graph proof. Its maximum is 4.221 ms and
publication-to-completion maximum is 5.087 ms, under the unchanged 40 ms limit.
This is correctness timing, not a real-time or sustained tracking qualification.

## Use and reproduction

From a source checkout with the three compiled, reviewed FJOBs:

~~~python
from framenpu_cdsp import Job
from framenpu_cdsp.research.convolution import load_landmark_graph
from framenpu_cdsp.research.landmark import StreamingLandmark

graph = load_landmark_graph("model/hand.ncnn.param", "model/hand.ncnn.bin")
jobs = {kind: Job.load(f"build/landmark_{kind}.fjob") for kind in ("hmx", "hvx", "sigmoid")}
with StreamingLandmark(graph, jobs) as executor:
    outputs = executor.run(chw)  # 3x224x224 replicated grayscale /255, flat CHW
# outputs["out0"], ["out1"], ["out2"], ["out3"] are FP32 arrays.
~~~

Build the pinned ncnn CPU adapters and private step-3 corpus as described in
[CONVOLUTION.md](CONVOLUTION.md), then export all graph references:

~~~sh
python tools/prepare_landmark_reference.py --model /path/private/model \
  --convolution-corpus /path/private/convolution-corpus \
  --reference /path/private/reference/ncnn_reference --output /path/private/landmark-corpus
framenpu-cdsp build-job tests/landmark_hmx.c --hmx --output build/landmark_hmx.fjob
framenpu-cdsp build-job tests/landmark_hvx.c --hmx --output build/landmark_hvx.fjob
framenpu-cdsp build-job tests/landmark_sigmoid.c --hmx --output build/landmark_sigmoid.fjob
~~~

Only after verifying the supported firmware,
idle gate/callback, original dispatch/code prefix and inactive HMX configuration:

~~~sh
sudo python tests/hardware_landmark.py --run-on-device --jobs build --model model \
  --corpus landmark-corpus --crop calibration_controller --stop-after conv_16 \
  --controls --output /tmp/landmark-pilot.json
sudo python tests/hardware_landmark.py --run-on-device --jobs build --model model \
  --corpus landmark-corpus --split calibration --controls --output /tmp/landmark-calibration.json
sudo python tests/hardware_landmark.py --run-on-device --jobs build --model model \
  --corpus landmark-corpus --split heldout --calibration-report /tmp/landmark-calibration.json \
  --output /tmp/landmark-heldout.json
~~~

The runner needs optional NumPy; the library and offline suite do not.
Reports retain per-layer/chunk timing and numerical statistics. Weight prepack,
patch packing, CPU reference work, upload, DSP wrapper, power close, restoration
and executor close are measured separately. Raw model/crop/output tensors stay
in private working directories. A failure stops immediately, with no retry,
automatic gate clearing or power recovery. See the [failure contract](ABI.md).
