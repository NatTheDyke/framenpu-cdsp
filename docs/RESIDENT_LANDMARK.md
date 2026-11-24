# Pinned resident landmark execution

`framenpu_cdsp.research.resident_landmark.ResidentLandmark` keeps the pinned v20a graph's
weights and intermediate tensors in the verified 6,152,192-byte DDR arena.
It uploads weights once per session, streams one FP32 input per frame and
returns the four heads. Packing and every model operation execute on DSP;
qualification reads resident tensors without feeding CPU results back into
the model. General ncnn import and arbitrary resident schedules remain unsupported.

## Storage and arithmetic

| Resource | Arena offset / bytes |
|---|---|
| Saved prefix and ownership/control | `0x1a22000`, 8,192 |
| Immutable 63-record operator program | `0x1a24000`, 8,192 |
| Packed weights and FP32 biases | `0x1a26000`, 2,051,904 plus alignment/guard |
| FP32 activation storage | `0x1c1b000`, peak 4,014,080 |
| Alignment and unused headroom | 20,352 |
| Reusable packed-input page | `0x1ff4000`, 45,056 |
| End guard | `0x1fff000`, 4,096 |

The original model file is 2,046,732 bytes. Packing removes ncnn tags,
interleaves pairs of weight rows for HMX and pads four odd dense-output rows.
Every original coefficient and bias bit is preserved. Reduction depth is not
padded in DDR. Integer HVX FP32-to-FP16 conversion uses round-to-nearest-even,
including subnormals and overflow. The pinned depthwise weights are normal or
zero; the loader rejects subnormal depthwise coefficients.

HMX computes `weights[N,K] * input[K,positions]`, producing CHW rows directly.
Each transaction clears the accumulator, loads the complete reduction through
K=704 and stores once. A separate DSP postprocess applies FP32 bias and clip;
the two confidence heads then use the previously qualified sigmoid. Activations
remain FP32 between layers, with the same FP16 HMX input/store boundaries as
[the streamed executor](LANDMARK.md). No new tensor quantization is introduced.

The ordinary lifetime schedule would need 4,816,896 activation bytes and would
not fit beside the weights. Only `convdwclip_1` uses a specialized in-place
stride-two downsample: each channel's 56×56 output occupies the prefix of its
112×112 input plane. Row order cannot overwrite future reads. A separate
forward compaction then packs the 64 output channels. Independent offline
future-read and compaction proofs cover every vector; hardware qualification
checks compaction bit for bit. All other tensors use first-fit lifetimes and
26 whole-buffer aliases, including residual branches and all four live heads.

## Ownership, completion and teardown

One arena lease, one direct-dispatch owner and one HMX worker serialize the
session. The arena lease/refusal record are shared with `ResidentGemm`;
a disappeared process/socket never permits adopting marked storage. Native
requests check nonce, immutable identity, monotonic sequence, phase and progress.
The host checks the immutable program and exact expected native state under
every request's gate, then checks metadata, workers/SSR/PC, raw nonfinite flags,
guards and optional numerical results before reclamation.

Initial reconciliation cleans/invalidates the previous cached data alias in
at most 32 KiB per real submission and saves effective original bytes after
completion. Subsequent model data accesses use uncached aliases. Code also
defaults to uncached; the opt-in [resident cache entry](RESIDENT_CACHE.md)
adds checked instruction publication and cleanup on normal and guarded-fault
returns, within the same reservations and timeout limits. The separate
[linked operator image](RESIDENT_LINK.md) executes from owned DDR instead and
lets one request cross layers.
Each native loop stops between whole work units after a five-millisecond tick
budget, with fixed unit caps. There is no mid-HMX cancellation. The public
4 KiB code slot, 8 KiB VTCM backup, 128 KiB borrowed job arena and 40 ms host
timeout remain unchanged. Eleven small modules are copied into the existing
code slot; persistent DDR is data storage, not executable code.

Close restores at most 32 KiB per real completed submission, walking backwards.
The ownership prefix retires only after body restoration and dispatcher removal;
a final independent readback checks every arena byte. Confirmed guarded faults
invalidate the frame and allow close only. Uncertain completion or a failed
checker retains code, power, arena and refusal record. No automatic gate reset,
power cleanup, replay or service restart is added.

## Numerical qualification and timing

The original resident baseline below is preserved for comparison. The subsequent
[gather optimization](DEPTHWISE_GATHER.md) reduces audited frame bodies to
183.6–187.1 seconds with bit-identical heads, unchanged limits and clean teardown.

Qualification uses the same pinned model, strict ncnn references, four recorded
infrared crops and frozen policy as [the streamed graph](LANDMARK.md). Policy
SHA256 is `381a033d6f4c10047998dfe42abb8cd6b18a756741c8a21690a3e6d6dc951f7b`.
Every layer passes both a local calculation from actual DSP inputs and propagated
maximum/RMS limits against ncnn. Heldout testing requires completed calibration
with identical source, native jobs, corpus manifest and policy hashes.

The final matrix passes 308 layer checks, 360 blob checks and 20,351,360
layer-output values across four crops. All 512 head values pass independent
saved-tensor readback and are bit-identical to the qualified streamed outputs.
The maximum local tolerance ratio is .80855; the maximum propagated RMS ratio
is .89798. Two exclusive sessions each run two different inputs with one weight
upload. All 93,741 main restoration audits pass, and every byte of the full
6,152,192-byte arena restores independently after both sessions. The DSP service
keeps its PID with zero restarts throughout the final matrix.

Supplemental checks pass 384,513 exact FP32-to-FP16 cases, 1,512 gather cases,
native metadata rejections, competing-owner refusal and guarded faults after
one HMX transaction and one in-place depthwise unit. Both faulted sessions
refuse reuse and restore the full arena on close. The ordinary SDK numerical
regression adds 72 restoration audits, including 38 HMX jobs. Offline tests
also verify interrupted-close retention and refusal after a lost lease.

An initial depthwise attempt exposed the full-length `vsetq(128)` mask clearing
its vector. The numerical checker retained resources and stopped; read-only
evidence was captured before an explicitly authorized reboot. Skipping the mask
for a full vector fixed the issue. No numerical limit or timeout was changed.

The original audited frames take **325.8–326.6 seconds**, with 23,209–23,215 requests each.
This is slower than the 71–79-second streamed baseline. Direct uncached
depthwise gathering dominates native time and command count; per-request power
setup/cleanup adds substantial wall time. The baseline also submits packing,
HMX and postprocess separately. The next optimization should address those
measured costs while preserving this dataflow and the frozen numerical policy.
Residency alone does not establish useful frame latency, sustained tracking
coexistence, thermal behavior or broader accuracy.

Each session uploads 2,051,904 weight bytes, or 2,060,288 immutable bytes including
the program/alignment/guard. Each frame uploads 602,112 input bytes and reads
512 final-output bytes. Qualification additionally reads intermediate tensors;
ordinary execution omits those reads. Initial load/reconciliation takes about
2.4 seconds and bounded teardown 1.5–1.6 seconds, separately from the frame body.

Metrics separate host plan/weight packing, immutable/input upload bytes and time,
load/reconciliation, per-module packing/arithmetic wrapper time, qualification
setup/readback, frame wall time and close. Audited/qualified runs include checks;
they must not be described as production inference latency. Sanitized results
are in [resident-landmark-validation.json](../evidence/resident-landmark-validation.json).

## Reproduction

Build modules on a workstation with the recorded LLVM toolchain:

```sh
python tools/build_resident_landmark.py --output /path/to/jobs
python tools/build_kernels.py --source tests/resident_conversion.c --hmx --output /path/to/conversion.fjob
python tools/build_kernels.py --source tests/resident_gather_probe.c --hmx --output /path/to/gather.fjob
```

Prepare private model/corpus references as in [LANDMARK.md](LANDMARK.md). The
hardware tools require explicit opt-in and root on the supported Frame:

```sh
sudo .venv/bin/python tests/hardware_resident_conversion.py --run-on-device --probe /path/to/conversion.fjob --output /path/to/conversion.json
sudo .venv/bin/python tests/hardware_resident_gather.py --run-on-device --probe /path/to/gather.fjob --output /path/to/gather.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --jobs /path/to/jobs --model /private/model --corpus /private/corpus --load-only --controls --output /path/to/load.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --jobs /path/to/jobs --model /private/model --corpus /private/corpus --controls --output /path/to/calibration.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --jobs /path/to/jobs --model /private/model --corpus /private/corpus --split heldout --calibration-report /path/to/calibration.json --output /path/to/heldout.json
```

Diagnostic `--crop`, `--stop-after` and `--fault-at LAYER KIND` runs cannot
qualify the graph. A session can also be used without qualification readbacks:

```python
from pathlib import Path
from framenpu_cdsp import Job
from framenpu_cdsp.research.convolution import load_landmark_graph
from framenpu_cdsp.research.resident_landmark import ResidentLandmark, JOB_KINDS

graph = load_landmark_graph("hand.ncnn.param", "hand.ncnn.bin")
jobs = {name: Job.load(Path("jobs") / f"resident_graph_{name}.fjob") for name in JOB_KINDS.values()}
with ResidentLandmark(graph, jobs) as session:
    first = session.run(first_chw)  # 3*224*224 finite FP32 values
    second = session.run(second_chw)
```

Model weights, recordings, private references and raw device dumps are not
bundled. The external model remains CC BY-NC 4.0; library code remains MIT.
