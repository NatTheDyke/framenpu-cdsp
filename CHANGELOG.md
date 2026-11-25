# Changelog

## framenpu-cdsp 0.1.1

- The exception-vector stubs of the SDK entry and of the Mercury loader no longer overwrite `r0`.
  EVB is shared by every hardware thread, and the firmware handler saves the interrupted thread's
  registers only from its own first packet on. During the dispatcher install/remove job the
  firmware's main thread holds its 50 ms acknowledgment deadline in `r1:0`, so an interrupt there
  could expire it at once ("stock completion window expired after DSP completion"). Each stub is
  now one extended PC-relative jump, and `tests/test_vector_stubs.py` decodes the shipped payloads.
  New `runtime.bin`, `dispatch_patch.bin`, loader (`56cbe683…`) and example job; only the stubs and
  the addresses after them differ. `dispatch.bin` and the Mercury image (1bcbbdd8) are unchanged.
  On a Frame (2026-11-24): 628 install/remove jobs with no false timeout, and the old and new loader
  give identical Mercury outputs (480 requests on random crops). The old stubs were not run as a
  control; 4 earlier faults in about 300 such jobs are the comparison.

## framenpu-cdsp 0.1.0

- framenpu's CDSP device backend as its own package, split out of framenpu 0.3.0 with its history.
  It depends on `framenpu>=0.4,<0.5` and registers the `cdsp` plugin (ABI 1). The bundled DSP
  payloads are unchanged (Mercury image 1bcbbdd8, which rebuilds byte-identical from this tree).
- The early job and matrix API and the CLI moved here: `framenpu_cdsp.Device`, `Job`, `Model`,
  ..., and the `framenpu-cdsp` command (`info`, `build-job`, `inspect-model`, `run-model`,
  `plan-resident`, `inspect-ncnn`). The experiments are `framenpu_cdsp.research.*`.

Earlier entries are framenpu's, from when both were one package.

## 0.3.0

- The device backend is now a plugin: `framenpu_cdsp` registers `CDSPPlugin` under the
  `framenpu.devices` entry point and framenpu finds it with `framenpu.devices.load()`
  (`FRAMENPU_DEVICES=name=module:attr` for source trees). Its `capabilities()` come from the
  image record; `compile()` plans each model (a `Refusal` when the image cannot run it) and
  `open()` returns the lease. MercurySession, `framenpu.open` and `framenpu serve` no longer
  import the backend: the plan reports where the graph divides between host and device
  (`split`), and the kernel's layout (stem columns, grid packing, staging, readback) moved to
  `framenpu_cdsp.staging`. framenpu's native host path keeps only the model kernels (inputs,
  FP16, hint, pooled heads) and binds to the staging workspace's buffers, so a request runs the
  same kernel calls in the same order. Plans and outputs are bit-identical to 0.2.0's (lane
  simulator, both host paths).
- The job-scoped AXI-master raise is a plugin setting: `CDSPPlugin(axi_mhz=748.8)`, or
  `FNPU_AXI_MHZ=748.8` in the service's environment (no framenpu option). An unknown rate is
  refused when the plugin loads; `capabilities().features` then lists `axi`, and each device
  session gets its own clock owner. Still off by default.
- The HMX clock rate is a plugin setting too: `CDSPPlugin(hmx_mhz=499.2)` or `FNPU_HMX_MHZ=499.2`
  (default 192 MHz). Each device session's power owner runs at the plugin's rate; an unknown rate
  is refused at load, and a raised one is listed as `hmx_fast`.
- The students' weights are on Hugging Face (`natthedyke1/mercury-ft7`,
  `natthedyke1/mercury-stereo3d-tpc1`, CC BY-NC-SA 4.0); their manifests record the repo and
  commit as `source`. `tools/fetch_models.py DIR` fetches every bundled manifest's weights by hash
  (Mercury's own models from Monado's repository, which framenpu does not re-host) and prints the
  model-gated tests' variables with `--env`.
- `tests/sim_plugin.py` is a `cdsp` plugin with no DSP: the real `CDSPPlugin` on the lane
  simulator, selected with `FRAMENPU_DEVICES=cdsp=sim_plugin:SimPlugin`, so `framenpu.open(...,
  "device")` and `framenpu serve` run their device path in CI and return the FP16 contract bit
  for bit. CI gains a `models` job (the whole suite with the fetched weights). The simulator
  now restores the job arena after each job, as the device does.

## 0.2.0

- `framenpu_cdsp` ships the qualified Mercury image (1bcbbdd8: streamed, 4 crops per request) in
  `data/mercury`; `framenpu serve` and `framenpu.open(..., "device")` use it when no image folder
  is given (`--jobs` is now optional). The repository check verifies it against its link record.
- `ping` reports each served model's manifest: the bundled manifest name matching the weights'
  SHA-256, the hash and the crops per request (`Client.manifests`). Wire protocol still version 4.

## Unreleased

- `framenpu.open(manifests, backend=...)` opens models by manifest name on the CPU
  (onnxruntime), the device or the NPU service, behind the CPU fallback latch
  (`framenpu.session`). The inference backends and the NPU service moved here from
  steam-frame-hand-tracking as `framenpu.backends` and `framenpu.service`; the service runs as
  `framenpu serve` with the same wire protocol (version 4). `framenpu/manifests` ships manifests
  for ft7, Mercury's keypoint model, the stereo student and the detector (weights not included);
  `open` also takes a weights file and finds its manifest by hash. Sessions add `infer_many`
  and `backend_of`; a model the service does not serve runs on the CPU without moving the
  others, and CPU models load on first use. New extra: `framenpu[cpu]` (onnxruntime).

- The device backend now lives in its own package, `framenpu_cdsp`, in the
  same distribution: transport, power and clocks, dispatch, residency, the
  Mercury kernel ABI planner, the DSP payloads and the SDK. The old
  `framenpu.<module>` names are aliases of the same modules, so imports,
  exception identity and test patches are unchanged. The Mercury session is
  split in two: `framenpu_cdsp.mercury_device.MercuryLease` validates the image
  and owns the arena, seal, staging, submission and completion checks;
  `framenpu.mercury_runtime.MercurySession` keeps model loading, planning, host
  work and scheduling. Image features now come from the link record instead of
  membership tests on the pinned hashes. The shipping DSP kernel sources moved
  from `tests/` to `kernels/`. All six pinned 20 KiB images and their loader
  rebuild byte for byte. The host C library is split the same way:
  `framenpu/host/fnpu_host.c` (model kernels, ABI 5) and
  `framenpu_cdsp/host/fnpu_layout.c` (window reads, stem columns, grid pack and
  unpack; `framenpu_cdsp._layout_native`), so the device backend no longer
  imports `framenpu`.

- The library split has started: [CONTRACTS.md](https://github.com/NatTheDyke/framenpu/blob/main/docs/CONTRACTS.md) specifies
  the four interfaces between framehands, framenpu and the framenpu-cdsp device
  backend (session API, model manifest, device plugin ABI, service protocol).
  `framenpu.contracts` gives the session and plugin protocols and a manifest
  validator, using only the standard library. `tests/test_boundaries.py` assigns
  every module to its future library and fails on any new import across those
  lines; today's ten crossings are listed and may only shrink. Nothing at run
  time changes.

- The resident Mercury image is now `5d15e715…` with loader `cc81fb7e…`, built
  with `--hmx-inline --pool --tiles --post-batch --pw-deep --pw-lg 3 --code-kib
  20`. Workers 1 and 2 run the depthwise, stride-2 and 1x1 post work while
  worker 0 drives HMX. Tensors that only 1x1 ops read are kept in B-tile
  layout, so those ops no longer gather their inputs. Posts are batched four
  vectors at a time. The full-resolution 1x1 ops run 8-block groups in a larger
  scratch that the planner flags in descriptor word 15 (bit 8). The image
  reservation grows from 16 to 20 KiB (0x1ff0000..0x1ff5000, seal vector
  0x1ff4f80); the runtime reads the size from `link.json` and accepts only 16
  or 20 KiB. DSP time per crop on the Frame fell from 2.96 to 1.63 ms, with
  bit-identical outputs. In a live A/B/A/B the DSP body of a 1-crop request
  fell from about 3.0 to 1.62 ms and the service's handling time from 5.1 to
  3.7 ms.

- The resident Mercury image (`9160e4c7…`) is now built with `--omit-fp
  --pw-pipe`. Without frame pointers, the 1x1 op's helper calls stop storing
  frames to the stack in cached DDR, and the barrier after every eight tiles no
  longer waits about 1.9 µs for those stores to finish. With the stall gone,
  the overlapped gather and post loops of `--pw-pipe` pay off. DSP time per
  crop on the Frame fell from 3.35 to 2.96 ms (1x1 ops 1.49 to 1.10 ms), with
  bit-identical outputs on 192 requests. In a live A/B/A/B the service's DSP
  wait per request fell from 3.95 to 3.46 ms. The loader is unchanged.

- Mercury parameter images can stay resident in VTCM between requests
  (`build_teacher_link.py --resident`, image `265fc6f4…`), in the FeatureNN
  model-blob region. The firmware reloads that region before FeatureNN's next
  use once its flag is cleared. DSP time per crop on the Frame fell from 4.66
  to 3.35 ms with bit-identical outputs, and the live NPU request median from
  10.8 to 9.4 ms. `MercurySession(resident=False)` keeps the copy-per-crop
  plan on the same image.

- The per-request HMX power check reads its 91 registers natively, on
  read-only mappings, with the same 32-bit accesses in the same order. It is
  used only after its first read matches the Python read. The check fell from
  159 to 81 µs (the rest is uncached register latency), and CPU per crop from
  1.32 to 1.25 ms (1 crop) and 1.02 to 0.98 ms (2 crops).

- The native host path binds its buffers, pointers and kernel arguments once
  per session and batch size. CPU per crop on the Frame fell from 1.51 to
  1.33 ms (1 crop), 1.19 to 1.03 ms (2 crops) and 1.06 to 0.91 ms (3 crops),
  with unchanged outputs.

- The native host kernels now process all crops of a request in one pass over
  each weight matrix, with bit-identical results. CPU per crop on the Frame
  fell from 1.24 to 1.17 ms for 2-crop requests and 1.16 to 1.04 ms for
  3-crop requests.

- Added a [native host path](docs/MERCURY.md#native-host-path) for Mercury
  requests: a freestanding C library (bundled for aarch64, hash-pinned,
  reproducible) for input checks, staging, readback and the host heads. It
  matches the NumPy path (staging byte-identical, one FP32-BLAS layer within
  2.4e-7), and session CPU per crop on the Frame fell by about 0.2 ms
  (1.76 → 1.56 ms for 1-crop requests, 1.40 → 1.17 ms for 3-crop requests).

- Moved the [Mercury heatmap head onto the DSP](docs/MERCURY.md#step-2c-heatmap-head-on-the-dsp)
  (`--mheads` image `582d79ce…`): two new descriptor kinds, regrid and 3x3 on
  HMX. On 96 real crops the outputs match the current image to FP16 rounding.
  Host CPU per crop fell from 4.5 to 2.3 ms, and session wall time from 10.6
  to 8.0 ms, with DSP body time 4.64 ms. `MercurySession` accepts either image.

- The tracker fork now [defaults to cores 5–6](docs/MERCURY_LIVE.md#heat-and-cpu-placement).
  A follow-up session showed the frequency cap adds little on those cores
  (about 0.2 W saved, 2–3 ms slower), and a 5-minute run stayed steady.
  Compositor frame timing under the new placement is not measured yet.

- Traced live thermal peaks to [shared CPU cores](docs/MERCURY_LIVE.md#heat-and-cpu-placement):
  Valve's XRService, vrserver and compositor run on the same cores 0–4 as the
  tracker and keep that cluster at 3.15 GHz. The tracker gained an opt-in
  `--uclamp-max` frequency cap; in short alternating trials, cores 5–6 with
  `--uclamp-max 512` lowered the hottest core p95 from 87–90 °C to 77–80 °C
  at about the same total CPU power and lower capture-to-collect latency.
  The live comparison tool now records rail power (MAX34417), per-zone
  temperatures and fan speed per run, and accepts a per-run frequency cap.

- Added [fresh-frame scheduling and camera-idle power release](docs/MERCURY_LIVE.md#fresh-frame-scheduling-and-idle-release)
  to the tracker integration. Each capture is submitted at most once, held
  HMX power is released after 0.2 s of blocked admission (camera stall, no
  tracked hand, thermal pause, unplaceable crop), and `--merc-fresh` submits
  right after optical flow. In alternating live runs (3 per scheduler),
  median frame age at submit fell 35.5 → 27.5 ms and capture-to-collect
  72.6 → 62.6 ms (p95 101.2 → 86.5 ms) at an unchanged accepted rate.
  Telemetry now splits request age by stage; the live report and comparison
  tools gained scheduler variants, start-temperature cooldown and a run
  deadline.

- Ran a [live stereo-camera comparison](docs/MERCURY_LIVE.md) of the integrated
  tracker. Two-crop processing wall time falls 34.09 → 31.23 ms (8.4%), and
  Mercury worker thread CPU time falls 29.66 → 17.76 ms (40.1%). Single-crop
  requests are slightly slower; the short CPU/NPU/NPU/CPU trials do not show a
  consistent overall tracking-throughput gain. Added paired-input and pipeline
  benchmark/report tools, explicit backend/DSP telemetry, and a Mercury fix
  for local pose tracking with `--no-bridge`. CPU remains the default.

- Added an installable `framenpu.mercury_runtime.MercurySession` and a
  [tracker integration](docs/MERCURY_INTEGRATION.md) for spyjacob's
  steam-frame-hand-tracking. Mercury inference is pluggable with CPU as the
  default and fallback. Available crops submit immediately in groups of 1–3,
  without padding or waiting for more observations. The session holds HMX
  power while tracking, releases it at idle, and preserves refusal state on
  uncertain completion. The packaged session passed 16 mixed-count requests
  on the Frame, with feature tensors bit-identical to replay qualification
  and the arena, power and service restored.

- Added [1–3 crop replay requests](docs/MERCURY.md#one-to-three-crops-per-request)
  (`build_teacher_link.py --replay`, frozen image; earlier images unchanged).
  One 55-descriptor table runs the actual crop count, at most 3, without
  padding. The harness mixes counts across rounds, validates per-crop end
  ticks and reports body time by crop count. Frame pilot and full run pass:
  4.98 / 9.95 / 14.92 ms DSP body for 1 / 2 / 3 crops, all crops bit-identical
  to qualification and all 16 final hashes unchanged. Four crops, misaligned
  stride and out-of-window stride are rejected; all restoration checks pass.

- Added a [grouped 1x1](docs/MERCURY.md#grouped-1x1)
  (`build_teacher_link.py --pw-group 8`, new image; existing images
  byte-identical). It runs the HMX tiles of 8 column blocks before posting
  them. The unexplained 1x1 post-loop wait turned out to be HMX execution
  (about 0.21 µs per tile load), now paid once per group. Mercury 1x1 drops
  from 3.55 to 3.18 ms and the request from 5.30 to 4.92 ms per crop,
  bit-identical on all 16 cases.
- Added a [fast depthwise](docs/MERCURY.md#fast-depthwise)
  (`build_teacher_link.py --dw-fast`, new image; existing images
  byte-identical and frozen in tests). It unrolls the 3x3 taps, inlines the
  post, keeps the overflow flags in a register per op, and keeps the FP16
  rounding separate with a register fence. Mercury depthwise drops from 1.54
  to 0.98 ms and the whole request from 6.02 to 5.30 ms, bit-identical on
  all 16 cases. Also documents the negative results: copy overlap, flag-slot
  placement, and store-drain probes.
- [Host time](docs/MERCURY.md#host-time-held-hmx-power-and-2-crop-batches):
  the Mercury harness records host phases per frame. `--held-power` keeps HMX
  powered for the session through the existing held-power owner, instead of
  a power cycle per request (about 5.9 ms). `--batch 2` runs two crops per
  request, sharing the parameter images (`build_plan(..., batch=2)`). Per-crop
  host wall falls from 14.1 to 7.8 ms against 6.0 ms on the DSP. Final
  tensors are bit-identical on all 16 cases.
- Added a [fast 1x1 post](docs/MERCURY.md#fast-1x1-post)
  (`build_teacher_link.py --pw-fast`, new image; existing images byte-identical
  and frozen in tests). It keeps the overflow flags in a register instead of
  a read-modify-write per output vector next to the HMX output page, which a
  new phase profiler (`--pw-profile`) showed to be the main 1x1 cost. Mercury
  1x1 layers drop from 8.77 to 3.64 ms, and the whole request from 11.15 to
  6.03 ms. Final tensors are bit-identical on all 16 cases.
- Added a [direct stride-2 depthwise](docs/MERCURY.md#direct-stride-2-depthwise)
  (`build_teacher_link.py --s2-direct`, new image; existing images
  byte-identical). It computes only the kept outputs, dealing input rows into
  even and odd columns, instead of a full stride-1 pass plus scalar
  decimation. The Mercury network's 8 stride-2 ops drop from 4.67 to 0.68 ms
  (`Conv_14` 8.3x) and the whole request from 15.14 to 11.15 ms. Final tensors
  are bit-identical to the previous image on all 16 cases, with every check
  and control passing.
- Ran the [whole Mercury convolution network on the DSP](docs/MERCURY.md#step-2b-the-whole-convolution-network-on-the-dsp)
  (step 2b): `Conv_0 .. Conv_139` as one 55-descriptor request on the same
  image. The stem runs as an HMX pointwise over host im2col columns, and the
  64x64 expansion is banded by channel groups instead of rows. 15.14 ms DSP
  body per crop. All 16 cases stay within the per-op FP16 bound (worst 0.39),
  and merged frames are bit-identical to per-op qualification. Against FP32,
  keypoints drift 0.185 px median / 0.669 px max with no `is_hand` flips. The
  first pilot stopped on a depthwise bound failure (1.26). The cause is the
  kernel's qf16 accumulation flushing sub-2^-14 products, and the depthwise
  bound now carries that absolute term (18·2^-14).
- Ran the [Mercury keypoint backbone on the DSP](docs/MERCURY.md#step-2a-backbone-on-the-dsp)
  (step 2a): `Conv_19 .. Conv_139` (36 ops, 32x32 to 160x8x8) as one
  resident request through the linked image, with an even-phase stride-2
  build option (`--s2-even`, new image only; teacher images unchanged) and a
  zero-copy hint concat. 6.58 ms DSP body per crop (forecast 8.59 ms). All
  16 private cases stay within the per-op FP16 bound (worst 0.73); merged
  frames are bit-identical to per-op qualification; keypoints with host heads
  drift 0.147 px median / 0.491 px max from FP32, with no `is_hand` flips.
  All 677 restoration audits pass. The 64x64 prefix still runs on the host.
- Started the [Mercury keypoint](docs/MERCURY.md) port (offline step 1): a
  dependency-free ONNX reader, a validated lowering of Monado's pinned
  `grayscale_keypoint_jan18.onnx` (FP32 matches ONNX Runtime within 7.6e-6),
  an FP16 HMX/HVX contract model (keypoint p95 0.46 px of a 128 px crop, no
  `is_hand` acceptance flips) and a teacher-calibrated DSP forecast (about
  19 ms per crop with heads on the host). No weights are redistributed; no
  DSP code or device run.
- Evaluated [128/160/192 crop inputs](docs/TEACHER_RESOLUTION.md) against the
  unchanged strict FP32 224 teacher. On the only presence-positive crop,
  128 input lowers presence from 0.999938 to 0.255875 and shifts landmarks
  by 28.50 px mean. Added a CPU-only reproducible runner, resampling control,
  statistics and tests. These are output-drift measurements without labeled
  ground truth; the DSP implementation and default input resolution are unchanged.
- Added optional [single-plane 224x224 grayscale input](docs/TEACHER_GRAYSCALE.md)
  via `--stem-gray`. Input transfer falls from 301,056 to 100,352 bytes with
  byte-identical prefix outputs and heads. Paired DSP prefix time falls from
  10.574 to 10.419 ms; host-wall medians remain effectively unchanged at
  41.81 ms. All 677 restoration audits pass. Original weights and RGB vector
  machine code remain unchanged; filter-folding limits are documented.
- Vectorized the banded prefix's max-pool and stride-2 output stitching
  behind `--stem-vector` ([results](docs/TEACHER_STEM_VECTOR.md)). Their combined
  time falls from 5.215 to 0.056 ms. Paired full-frame prefix time falls from
  15.676 to 10.559 ms, and host wall from 47.942 to 42.201 ms. Outputs remain
  byte-identical; all 992 restoration audits pass. The scalar image remains
  byte-identical, and the vector image fits the same 10,800-byte footprint.
- Added an optional [DSP prefix profiler](docs/TEACHER_STEM_PROFILE.md) with
  per-operator, HMX/fence, depthwise, halo and movement measurements. Scalar
  pooling/stitching and decimation consume 5.21 ms; the two depthwise
  operators consume 2.06 ms. Baseline machine code and all profiled outputs
  remain unchanged; 800 restoration audits pass.
- Added a paired [RGB-to-head prefix comparison](docs/TEACHER_PREFIX_COMPARISON.md).
  A native ncnn FP32 CPU prefix plus DSP tail measures 29.02 ms host wall versus
  47.51 ms for the two-request DSP path on the same pinned host core. Both
  pass the existing four-crop accuracy ceilings; the CPU FP16 hybrid fails
  one crop's RMS ceiling. Kernel code is unchanged.
- Added a qualified [banded RGB prefix](docs/TEACHER_STEM.md), including the
  max-pool skip branch, feeding the resident tail and heads in two requests.
  Prefix depthwise reductions and original biases use QFloat32 arithmetic
  with FP16 tensor storage. Median DSP time is 15.71 + 16.73 ms; complete
  normalized-RGB-to-head host wall is 56.75 ms. All frozen accuracy ceilings,
  384 restoration audits and 276 offline tests pass.
- Added optional resident global-average pooling and four FP32 heads
  ([TEACHER_HEADS.md](docs/TEACHER_HEADS.md)). The qualified tail stays in one
  request and returns 512 bytes; DSP time is 16.74 ms. Host wall through all
  outputs is 29.26 ms versus 37.91 ms for the current Python host-head path.
  All frozen accuracy ceilings and 238 restoration audits pass. The 112x112
  prefix remains outside this implementation.
- Fixed CI's NumPy dependency handling: numerical tests run with NumPy in the
  Python matrix, and the layer-chain check skips when it is absent locally.
  Applied the repository's formatting rules and fixed two lint failures.
- Restored selectable four-block coefficient sharing in the linked teacher
  depthwise loop ([TEACHER_DW4.md](docs/TEACHER_DW4.md)). Hardware outputs remain
  bit-identical, but the median tail is 15.85 ms versus 15.77 ms for two blocks
  on the same boot, so two blocks remains the default. Both variants pass all
  controls and 719 restoration audits in total. Added the static operation
  count and addressing assessment for sampled-only stride-2 row tiles.
- Added an optional resident stride-2 teacher-tail path
  ([TEACHER_STRIDE2.md](docs/TEACHER_STRIDE2.md)), qualified on hardware. It keeps
  all 42 compute ops in one 45-descriptor linked request, with an explicit
  two-request split option. The three stride-2 outputs stay in VTCM; the
  qualified per-layer kernels and default linked image remain byte-identical.
  All four corpus crops match qualification byte for byte. Median tail time
  is 15.80 ms in-DSP and 26.69 ms host wall (previous linked run: 18.80 ms
  and 76.85 ms). All controls and 359 restoration audits pass. The two-request
  option remains checked offline only.
- Teacher tail through a linked operator image ([TEACHER_LINK.md](docs/TEACHER_LINK.md)).
  The resident and stride-2 kernels are linked unchanged into one image,
  sealed in the owned 16 KiB reservation at 0x1ff0000. A 1,960-byte slot
  loader enters it without per-request publication. Per frame, in-DSP time
  drops from 30.0 ms to 18.8 ms, and per-job overhead from 1.6 ms to
  0.03-0.06 ms. Linked frames are bit-identical to qualification on all four
  corpus crops. Existing loader and kernel builds are byte-identical.
- Resident teacher tail ([TEACHER_RESIDENT.md](docs/TEACHER_RESIDENT.md)).
  One opcode-111 kernel runs whole sequences of 1x1 (HMX) and stride-1
  depthwise (HVX) layers, with tensors kept in VTCM, from a natively validated
  op table. The stride-2 layers stay as opcode-110 jobs. Per frame this is 7
  jobs instead of 42, 30.0 ms of in-DSP time instead of 96.2 ms, and
  94.6 ms of host wall time instead of 430 ms. Every compared tensor was
  bit-identical to the per-layer qualification run on all four corpus crops,
  and accuracy is unchanged.
- Ran the unchanged teacher in FP16 on the DSP from the 56x56 stage to the
  heads ([TEACHER_TAIL_FP16.md](docs/TEACHER_TAIL_FP16.md)): 42 jobs per crop
  (HMX 1x1 with fused bias, clip and residual; HVX QFloat16 depthwise, stride
  1 and 2) on the four corpus crops. Every check passed. Each layer stayed
  within its FP16 bound (worst ratio 0.49). The heads stayed within the frozen
  ceilings: landmarks within 0.385 px, presence within 2.8e-3. Intermediate
  tensors exceed the FP32-calibrated per-layer tolerances. Compute is 10.1 ms
  per crop; activation copies and the 42 job wrappers dominate the rest.
- Re-forecast the unchanged public teacher on HMX + HVX
  ([TEACHER_HMX_HVX.md](docs/TEACHER_HMX_HVX.md)). Measured single-thread HVX
  issue rates (1.84 GHz core; about 2 cycles per load, `valign` or QFloat16
  multiply+add) and two FP16 depthwise probes over all 12 stride-1 teacher
  depthwise layers. v2 (guard-column rows, pre-splatted coefficients, four
  blocks per coefficient load) takes 1.76 ms, versus 8.36 ms for v1 and about
  250 ms for the existing exact path. The teacher's convolution forecasts at
  about 7-10 ms with no retraining. Activation traffic in the 112x112 stage,
  job count, stride-2 depthwise and FP16 accuracy remain to be proven.
- Added the v3 layer-chain executor prototype
  ([HMX_SPATIAL_BLOCK.md](docs/HMX_SPATIAL_BLOCK.md#v3-layer-chain-executor)).
  It runs whole fused 3x3+1x1 layers from runtime descriptors, with weights,
  tables and activations in a window of the persistent DDR arena. Each layer
  writes the next layer's input on the device. The five-layer 14x14 chain
  takes 1.37 ms of kernel body and the four-layer 7x7 chain 2.13 ms, of
  which 80% is weight streaming from uncached DDR. Each extra job adds about
  1.6 ms of in-DSP wrapper time, so chains should run as few jobs as
  possible. All layers matched the oracle, fault and rejection controls
  passed, and the arena was restored byte for byte.
- Added the VTCM-resident v2 HMX spatial block. It stages weights, the source
  band and mid tiles in the proven 1,912 KiB FeatureNN scratch span. Steady
  per-block time falls 9-30x versus v1, and the half-width convolution
  forecast drops from about 99 ms to about 8 ms, inside the 12-20 ms gate.
  Not yet modeled: layer-level job splitting, inter-layer traffic, and the
  stride-2 and stem layers. HMX operands need more than 128 B alignment, so
  v2 aligns them to 4 KiB.
- Mapped VTCM ownership from the firmware and a read-only all-VTCM survey
  ([VTCM_MAP.md](docs/VTCM_MAP.md)). FeatureNN worker-0/1 scratch
  `0x31002000`-`0x311e0000` (1,912 KiB) is per-request scratch, serialized
  with direct jobs. An approved write canary passed: fill and verify, host
  pattern check, persistence across two sessions, and no change elsewhere
  in VTCM. Gaze and the FeatureNN model range stay excluded. Not yet used
  by any kernel; the headset was idle and hand tracking was not exercised.
- Added and measured a fused HMX spatial->projection block prototype
  (device gather, packed FP16, 8 KiB VTCM, one HMX owner) with per-phase
  timing and batching, residency, instrumentation and no-HMX controls. All
  numerical, fault and restoration checks pass. HMX costs 0.2-0.35 us per
  load; gathering dominates. The half-width forecast is about 99 ms of
  convolution, missing the 12-20 ms gate. Weight copies and barriers alone
  exceed it inside 8 KiB of VTCM.
- Added public-weight HMX model feasibility accounting and distinct-operand
  native probes. Packed streaming dominates the tested pipeline; a half-width
  fused 3x3 candidate forecasts 47 ms of prepacked convolution work and still
  needs reuse and training/accuracy validation. No new graph is integrated.
- Added bounded native arithmetic and real-weight depthwise experiments.
  Direct IEEE HVX variants return zeros on the tested configuration; supported
  persistent QFloat32 reductions and FP16 storage pass arithmetic/range and
  restoration checks on four crops. Isolated kernel gains are modest; no new
  arithmetic is integrated into the graph or qualified for tracking accuracy.
- Profiled fused resident host overhead without changing production execution:
  ordinary API wall time averages 762 ms, including 498 ms native, 150 ms
  Python state replay and 74 ms input checking/conversion/packing. All 48
  frames preserve qualified head bits and restore cleanly. Added an opt-in
  audited/ordinary profiling harness and three timing-accounting regressions.
- Omitted the unreachable bias/clip operator and its dispatch branch from
  fused convolution builds, reducing the linked image from 13,752 to 12,888
  bytes. Four crops preserve all head bits; fault, teardown and SDK checks
  pass. Non-fused images remain byte-identical.
- Added an opt-in fused 1x1 convolution path
  (`--dw-reuse 6 --dw-row-tile --conv-fast`): the HMX operator adds bias and
  clips while storing each row, removing the separate bias unit, and packing
  and HMX row handling keep no vector on the job stack. Convolution time falls
  from 497 to 182 ms, native time from 0.82 to 0.50 seconds and a frame from
  1.12–1.21 to 0.67–0.77 seconds (43 to 27 requests). Four crops keep
  bit-identical heads across 26,615 main audits; nine guarded faults, controls
  and the SDK regression pass. Default, reuse and tiled builds are
  byte-identical. Added six offline regressions.
- Added opt-in row tiling for the 7- and 14-wide depthwise layers
  (`--dw-reuse 6 --dw-row-tile`): four or two output rows per pass, still one
  row per work unit, with a 16 KiB operator reservation and its own loader for
  tiled builds only. Depthwise time falls from 421 to 291 ms and a frame from
  1.28–1.38 to 1.08–1.19 seconds (50 to 43 requests). Four crops keep
  bit-identical heads across 26,745 main audits, including frames interrupted
  at every offset inside a pass; four guarded faults, controls and the SDK
  regression pass. Default and reuse builds are byte-identical. Added seven
  offline regressions.
- Added opt-in linked depthwise broadcast-coefficient and product-mask reuse
  (`--dw-reuse 6`) within existing VTCM/code/budget limits. A sequential pilot
  reduces depthwise wrapper time 7.9% and frame time 6.1%, from 52 to 50
  requests. Four crops preserve all saved head bits under the frozen policy;
  1,571 main restoration audits, guarded faults, mask controls and ordinary
  SDK regression pass. Added five offline reuse/bounds regressions. Numeric
  format and tap accumulation order remain unchanged.

- Added opt-in session-held HMX power for the resident landmark executor
  (`hold_power=True`): one power-on under the operation gate at open, a
  read-only register proof under the gate before every request, one verified
  power-off at close. A linked frame falls from 1.66–1.69 to 1.36–1.39 seconds
  for the same 52 requests. Four crops keep bit-identical heads across 1,495
  main audits including 30-second idle holds; guarded faults and rejections
  pass. The default remains per-request power; coexistence with another HMX
  power user is not established. Four offline regressions and an idle-hold
  hardware check were added.
- Linked resident operators now read resident state, layer records and staged
  tensor rows from saved VTCM instead of word-by-word or unaligned from
  uncached DDR, keep no HVX register on the stack in per-unit paths, and take
  the input in one request. A frame falls from 1,532–1,533 requests and
  48.8–49.0 seconds to 52 requests and 1.66–1.69 seconds at the same 20 ms
  budget; native time falls from 30.8 to .99 seconds. Four crops keep
  bit-identical heads under the frozen policy across 1,494 main restoration
  audits; guarded faults, rejections and the uncached-alias control pass. The
  eleven-module build is byte-identical. Added a memory-access cost probe and
  its build tool.
- Linked the ten resident compute operators into one position-independent
  image in owned DDR, entered from a small slot loader, with requests that
  cross phase and layer boundaries up to a host tick budget or stop point.
  Seal/retire commands bracket cached instructions; the host replays each
  reported unit count through its independent state machine. Four crops retain
  identical heads and frozen limits across 13,299 main restoration audits.
  At a 20 ms budget, frames fall from 10,959–10,964 requests and 153.9–154.6
  seconds to 1,532–1,567 requests and 48.8–51.4 seconds. The eleven-module
  path and its module bytes are unchanged. Added a linked-build verifier and
  nine offline regressions. Remaining requests are bounded by depthwise
  native time.
- Added opt-in cached instructions for the pinned resident landmark executor,
  with uncached data, unchanged work/resource/timeout limits and one HMX owner.
  Versioned entry tags, selected-PC/fault checks and normal/fault cleanup markers
  are required before resource reclamation. All four crops retain identical
  heads and frozen error limits across 44,739 main restoration audits. Audited
  frames fall from 183.6–187.1 to 153.9–154.6 seconds (17.0% less time).
  Added linked-entry/relocation verification and seven offline regressions;
  public SDK payloads remain byte-identical. The default stays uncached.
- Optimized resident depthwise/max-pool gathers with HVX padding masks and
  bounded scalar tensor-edge copies. Four crops retain bit-identical heads
  under the unchanged numerical/ownership/timeout rules. Audited frames fall
  from about 326 to 184–187 seconds (1.76×); depthwise native time improves
  2.38× with 54.5% fewer requests. Added paired baseline/current bit-preservation,
  bounds and timing probes plus five offline regressions. The unchanged-policy
  graph matrix passes 57,800 restoration audits and full-arena teardown.
- Added a pinned resident landmark executor: weights upload once per session,
  FP32 intermediates stay in the verified DDR arena, and packing/operators run
  on DSP. Lossless weight-row packing and a proven in-place downsample plus
  compaction fit the graph without new quantization. All 77 layers, 90 blobs
  and four outputs pass four crops under the unchanged limits; all 512 head
  values are bit-identical to the streamed baseline. Two changing inputs per
  session pass 93,741 main restoration audits and full-arena teardown.
  Added bounded reverse restoration, persistent refusal/state checks and
  eleven offline regressions. This uncached baseline takes 325.8–326.6 seconds
  per audited crop; depthwise gathering and request overhead need optimization.
- Qualified an experimental pinned complete landmark graph executor: full-depth
  HMX convolutions/heads through K704, HVX depthwise/pooling/residual operations,
  native sigmoid and lifetime-aware tensor aliases. Intermediate tensors remain
  on the host. All 77 layers, 90 blobs and four outputs pass frozen ncnn/local
  checks on four recorded crops: 20,351,360 values and 25,455 restoration audits,
  including seven guarded traps and seven native rejections. Added exact zero-add/sigmoid
  identities, all-blob reference tooling and seven offline regressions. Debug
  failures retained resources and were followed by explicitly approved resets;
  the final matrix and ordinary SDK regression restore cleanly. Audited graph
  bodies take 71–79 seconds per crop; residency and real-time tracking remain open.
- Qualified real-weight `convclip_0`, pointwise `conv_10` and K=64 `conv_12`
  against hash-pinned ncnn/v20a references and recorded infrared crops.
  Added strict whole-graph/binary preflight, bounded patch lowering, native FP32
  bias/clip and frozen arithmetic plus total-error limits. All 2,207,744 values
  pass calibration/held-out checks and 3,558 restoration audits with no restart.
  Added external reference builders and 13 offline regressions; raw models/images
  stay private. This remains a slow layer-local proof, not full landmark inference.
- Qualified bounded resident 32×K×32 GEMM for K=32/64/128 with distinct operands:
  weights upload/prepack once per session, changing inputs, checked generations
  and descriptor bounds, cached execution and fault-safe cache teardown. All
  300 measured jobs plus warmups/reuse match 325,632 values; 21 rejections,
  three guarded faults and code replacements pass 372 restoration audits.
  Ordinary GEMM/XOR regressions add 78 passing audits with no service restart.
  Added separate host/DSP phase timings and 12 offline lifecycle regressions.
- Added offline resident-DDR storage planning with explicit tensor layouts,
  strides, aliases and graph-derived lifetimes. The CLI reports bounded aligned
  allocations, scratch costs and reuse for a residual example. Documented the
  proposed ownership, session, completion and cache contract; device execution
  remains future work. Added 21 offline regressions, including varied schedules.
- Ported nested-C-frame exception unwinding to the public SDK: reconstruct FP
  from the saved entry SP before deallocation. Rebuilt the 3,672-byte runtime,
  vector-add example and dispatch patch helper; existing custom FJOBs need a
  rebuild. Native qualification passes 48 guarded traps and 64 normal jobs
  across CPU/HMX modes and both dispatch paths, plus GEMM/XOR and multi-tile
  regressions. All 288 main-suite restoration audits pass without a restart.
- Qualified cached-code HMX execution at unchanged clock configuration:
  800 measured jobs and 32 warmups match exactly. Deep resident HMX p50 falls
  from 9.51 ms to 56.25 us; fitted load slopes are 1,264–1,269 GMAC/s. Fresh
  sessions still take about 33 ms; this is not full-model throughput.
- Qualified two-block loads in a separate uncached diagnostic: another
  800 measured jobs and 32 warmups pass, with about 2.52x deep-loop improvement.
  Activation/weight storage aliases; distinct-weight model execution is pending.
- Experimental entry cleans cached code on normal and guarded-fault return,
  with confirmation checked before host reclamation. Fixed its nested-C-frame
  unwind by restoring FP as well as SP; three audited cached traps pass.
  Added portable fixtures, sanitized evidence and clock-drift checks. The later
  public unwind port is qualified separately from these cached-code timings.
- Documented the verified 5.87 MiB persistent DDR region. Resident allocation
  and graph execution remain future work. Publication checks exclude local
  handoff/recovery files and scan the portable multi-worker builder in source
  archives.
- Numerically checked single-owner resident GEMM baseline: all output tiles
  match across 400 measured jobs and 16 warmups, with a 7.41–7.44 GMAC/s load
  slope. Added separate HMX, packing, copy, conversion and host-phase timings,
  repeated latency quantiles, reproducible fixtures and sanitized evidence.
- Full-tile accumulator isolation tests across workers 0/1/2: all isolated and
  serialized controls pass; deferred stores and concurrent transactions expose
  shared HMX state. The recovered clear/load/store sequence must stay under one
  owner for the complete transaction. Added bounded synchronization, phase
  timing, exact per-worker references and sanitized hardware evidence.
- HVX output unpacking and exact FP16-to-FP32 bit conversion for every supported
  GEMM output tile, retaining scalar FP32 accumulation order. The 128-byte row
  buffer stays inside existing VTCM scratch; the runtime is 3,672 bytes.
- Native validation covered all 65,536 FP16 conversion patterns, 71 seeded
  accumulation cases, numerical regressions and 573 restoration audits. Over
  100 warm direct-dispatch samples, two-depth-tile 32x64x32 publication-to-completion
  p50 fell from 15.82 to 6.99 ms against the previous input-packing runtime.
- HVX packing for aligned full FP16 tiles with row strides 32/64, plus aligned
  vector bulk copy with scalar tails. Padding and unaligned cases retain scalar
  paths. Hardware qualification passed 136 layout cases, 53 copy cases and the
  existing numerical suites; 310 submissions passed restoration audits.
- Against the previous runtime, two-depth-tile 32x64x32 publication-to-completion
  p50 fell from 25.28 to 15.79 ms over 100 warm direct-dispatch samples. The
  runtime remains within 4 KiB at 3,520 bytes.
- Opt-in `matmul(..., tiles_per_job=2)` on direct dispatch: two depth products
  per HMX job, ordered FP32 accumulator seeds and final-only bias/ReLU. A
  32x64x32 device comparison reduced four submissions to one and median wall
  time from 60.89 ms to 38.29 ms. The 40 ms execution timeout is unchanged.
- Context flag bit 1 identifies direct dispatch; the new multi-tile opcode
  rejects the benchmark path and validates its two-product bound on the DSP.

- Opt-in reversible direct dispatch for the pinned ABI 9 firmware, bypassing
  stock benchmark arithmetic and joining workers before master completion.
- Direct-compatible SDK entry metadata, worker-0 kernel timing counters and
  explicit empty/copy/GEMM comparison tooling.
- Host lifecycle/ownership failure tests and reproducible dispatcher/helper
  builds. CPU/DSP device validation passed, including 100 warm samples,
  metadata/client compatibility and firmware restoration.
- HMX qualification passed on both dispatch paths, with all five padded/tiled
  GEMMs and the saved XOR model matching their references. A new opt-in harness
  audited 78 correctness jobs (39 HMX jobs) and separately measured 100 warm
  samples per backend. Tracking coexistence remains pending; the default
  remains the benchmark backend.
- Correctly wait for master completion when direct workers return while the
  master still owns processing state 2.

## 0.1.0 — initial experimental release

- Python Device/Job APIs and a freestanding Hexagon v75 C SDK.
- Tiled HMX GEMM, DSP accumulation, bias/ReLU, clipping and broadcast add.
- Checked Dense/ReLU/Clip models and synthetic examples.
- Firmware guards, bounded jobs and explicit recovery states.
- Real-device custom-job, tiled-GEMM and serialized XOR validation.
- Standalone packaging, development docs, CI and provenance checks.

General FrameTop/ncnn import and real-time hand tracking remain unsupported.
