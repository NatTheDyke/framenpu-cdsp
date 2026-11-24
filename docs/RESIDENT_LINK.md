# Linked resident operators and cross-layer requests

`ResidentLandmark(graph, load_linked(folder), ...)` runs the pinned v20a graph
from one linked operator image instead of eleven modules copied into the code
slot one request at a time. A request may then cross phase and layer boundaries.
It ends between whole work units at a host tick budget, at a host stop point, at
frame completion or on a guarded fault. The eleven-module path is unchanged and
remains the default input; its modules still build to the same bytes.

Arithmetic, tensor layout, weights, the frozen numerical policy, the single HMX
owner, the 8 KiB VTCM backup, the 128 KiB borrowed job arena and the 40 ms host
timeout are unchanged. Tensors and weights stay in DDR on their uncached
aliases. What changes is where operator instructions live, how often they are
published, how many units one request may complete, and, since the
[working-copy revision](#working-copies-and-staged-data), how the operators
read that data.

## Two images

| Image | Where | Size | Contents |
|---|---|---:|---|
| Loader | existing 4 KiB code slot, copied per request as before | 3,836 bytes | entry/fault vectors and the admin commands |
| Operators | owned DDR `0x1ff0000`, 12 KiB reservation (16 KiB for [row-tiled builds](DEPTHWISE_TILE.md)) | 11,476 bytes | the ten compute operators and `rg_batch` |

The reservation lies in previously unused pages between the activation
high-water mark (`0x1fef000`) and the packed-input page (`0x1ff4000`), with one
unused page on each side. Both the offline plan and the native program check
bound tensors below it. The image has no writable data or BSS.

The operator image is a single translation unit that includes the existing
operator sources. A handful of macros in `resident_graph.h` select the loop
bound, budget test and return path; without `RG_LINKED`/`RG_LOADER` they expand
to the previous text, and the eleven cache-entry modules reproduce their
qualified hashes. `tools/check_resident_link.py` verifies the linked outputs:
the loader's vector table and forwarding stubs (same rules as the cache entry),
its dispatch constants, the image entry at offset zero, PC-relative C
relocations only, no indirect calls or jumps other than returns, no cache or
exception-return instructions, and no embedded address of the image itself.
The same image bytes therefore run through either instruction alias.

```sh
python tools/build_resident_landmark.py --linked --output /path/to/link-jobs
python tools/check_resident_link.py --jobs /path/to/link-jobs --output /path/to/link-verification.json
```

## Entering the image

The loader entry is the cache entry with staging, publication and per-request
cleanup removed; the slot image always executes uncached. Its dispatch tail-jumps
to the operator image only for opcode 32 and only while resident state word 31
holds the seal. Otherwise the request reaches the loader's own admin code, which
rejects it. Context parameter zero selects the uncached (`0xf1ff0000`) or cached
(`0x88ff0000`) instruction alias. Fault vectors stay in the slot, so a guarded
synchronous fault inside the image returns through the same uncached recovery
path as before.

Two new loader commands bracket the image's lifetime:

- **Seal (7).** Allowed only when the immutable program is loaded and no seal is
  set. The host writes the image under the request's gate; the loader discards
  data- and instruction-cache lines for the whole reservation through the cached
  alias, then sets the seal. The host reads the image back before accepting the
  completion.
- **Retire (8).** Clears the seal and discards the same cache lines. Restoration
  commands are refused natively while the seal is set, so no original byte is
  written back under live cached instructions. Close issues it first, including
  after a confirmed fault or an unfinished frame.

Nothing writes the reservation through the cached alias, so lines are
discarded rather than written back. Between seal and retire there is no
per-request publication: the cache-entry path's roughly 1.5 ms of staging and
cleanup per request is not paid. The host compares all 12 KiB with the sealed
image before and after every request while the seal is set, and binds the image
and loader into the session identity.

## Batch requests

A batch descriptor carries the starting layer and progress (checked natively
against resident state), a unit limit, a tick budget of at most 480,000
reference ticks (25 ms), and a stop point `(layer, phase)`. `rg_batch` validates
the descriptor once, accepts the sequence once, then repeatedly runs the
operator that resident state selects. Each operator still proves it is the
expected one and that its tensors lie inside owned storage. The loop ends when:

- the frame completes or a guarded fault is recorded;
- the unit limit is reached or the tick budget has elapsed, both tested only
  between whole work units;
- a phase ends exactly at the stop point.

The host does not trust the reported position. Every unit's operator follows
from resident state alone, so the reported unit count determines the result:
`_batch_transition` replays that many units through the same independent state
machine used for single-module requests and requires the native state to match
word for word. A count that runs past the stop point, past frame completion or
past a fault is rejected, as is any PC outside the image at the selected alias.
Rejected requests must leave state and sequence unchanged.

Qualification stops once per layer, reads that layer's complete output and
applies the same local and propagated checks as before. Two extra stops exist
where a later phase overwrites what must be read: the strided in-place
downsample before compaction, and the logit before each sigmoid. Unqualified
execution uses no stop point and runs each request to its budget.

`budget_ms` defaults to the previously qualified five milliseconds. Raising it
is an explicit choice; the completion window is still 40 ms and a missed window
still poisons the session.

## Working copies and staged data

A [cost probe](#access-costs) showed that the first linked image spent nearly
all of its 30.8 seconds per frame on three memory patterns, not on arithmetic.
The linked build now avoids them. None of this changes a tensor byte, a
weight, the order of any floating-point operation or the per-unit state
sequence; the eleven-module build compiles the same sources to its previous
bytes.

- **State and layer record.** `rg_batch` copies the 32 resident state words
  into the saved VTCM page and operators work on that copy. It is written back
  to DDR as one vector after every work unit, before every operator return and
  before a guarded trap, so DDR holds exactly the per-unit values the previous
  image wrote at each of those points. The current 128-byte layer record is
  copied once per layer; the program is immutable while a frame runs. The
  fault flag is read once per request.
- **Aligned loads.** An unaligned vector is taken from VTCM after copying the
  two aligned DDR blocks that cover it. Depthwise copies each in-range kernel
  row of a unit into VTCM with two to four aligned loads and takes every tap
  from there. Partial vectors, the strided stem gather and the 7×7 average
  pool read whole vectors and mask or transpose them instead of reading
  scalars. Bytes read beyond a tensor stay inside the owned arena.
- **Reduce vector spills.** Guarded traps leave the operator before calling.
  Some live packing vectors are parked in VTCM; the disassembly results below
  identify remaining stack accesses in pack, stem and HMX.
- **Bias and tap words.** One vector load serves the next 32 bias words and
  one serves a depthwise channel's kernel.
- **Input upload.** The host already wrote the input under each request's
  gate; the linked loader now accepts all 602,112 bytes in one request
  instead of 19.

All scratch lies in the existing saved and restored 8 KiB of VTCM.

### Access costs

`tests/resident_access_probe.c` times N repetitions of one primitive per
request and the host fits the slope between two repeat counts (19.2 MHz
reference ticks; empty loop subtracted). Owned DDR is only read, and
cached-alias lines are discarded before return.

| Access | Microseconds |
|---|---:|
| Scalar read, uncached DDR | 1.01–1.13 |
| Scalar read, cached alias (warm) / VTCM | .003 / .026 |
| Scalar write, uncached DDR | .003 |
| Vector load, DDR aligned (either alias, warm or cold) | .105 |
| Vector load, DDR unaligned at a fixed 4-byte skew | .216 |
| Vector load, DDR unaligned at overlapping 4-byte steps | 1.01 |
| Vector store, DDR aligned / unaligned | .104 / .13–.17 |
| Vector load or store, VTCM (aligned or not) | .002–.003 |
| Vector stored to the job stack and reloaded | 1.41 |
| Depthwise tap arithmetic on registers | .012 |

The cached data alias does not speed vector loads, so tensors were left where
they are. These are measurements on one device and firmware, not a model of
the memory system.


The completed native-time investigation adds stack, VTCM forwarding, alignment
and state-flush controls. The five-image probe passes 545 restoration audits.
Selected empty-loop-subtracted slopes (microseconds per iteration):

| Primitive | µs |
|---|---:|
| Scalar read from cached job stack / framed call with two leaf calls | .0007 / .0094 |
| VTCM vector store then reload / overlapping unaligned reload | .0174 / .0196 |
| Scalar VTCM write then vector load of the same vector | .0032 |
| Scalar VTCM write then vector copied to DDR / direct register-vector DDR store | .2072 / .2074 |
| Two aligned DDR loads joined by valign / staged through VTCM | .2763 / .3987 |
| Partial store through 17 scalar copies / masked vector store | .4288 / .1042 |

The scalar-write-to-vector-load fixture does not show the suspected microsecond
state-flush stall. The stack scalar/call and VTCM forwarding fixtures also do
not explain the remaining per-unit cost. These are timing controls; adopting
valign or masked stores requires separate bounds, arithmetic and model checks.

A fresh calibration crop passes all 77 layer checks, with all 128 saved head
values independently byte-identical to the held-power baseline. A following
frame takes 1.320 seconds and 52 requests. The phase profile's depthwise work
uses 81,760 units and 455.9 ms across 31 requests. Pack uses 39,392 units,
postprocessing 95,328, stem packing 7,168 and HMX 3,439. Their measured wrapper
totals include roughly 0.1 ms of fixed overhead per profiling request; the
profile deliberately makes thousands of small requests, so those totals must
not be added as production native time. The complete graph/profile session
passes 6,125 restoration audits and independent whole-arena restoration.

Disassembly corrects the earlier broad statement about stack spills: pack and
stem each still contain five vector stack stores and five loads; HMX contains
six stores and seven loads. Depthwise and postprocessing contain none. These
are static instruction counts, not dynamic counts per unit. Removing remaining
pack/stem vector spills is the first bounded optimization to try; fusing bias
and clip into HMX stores would then remove a separate postprocessing pass.
The native image is unchanged during this profiling run. Raw phase rows,
probe slopes/samples, hashes and caveats are preserved in
[native-time-profile.json](../evidence/native-time-profile.json).

```sh
python tools/build_access_probe.py --output /path/to/probe
sudo .venv/bin/python tests/hardware_access_probe.py --run-on-device --reads /path/to/probe/access_reads.fjob --stores /path/to/probe/access_stores.fjob --taps /path/to/probe/access_taps.fjob --stack /path/to/probe/access_stack.fjob --unit /path/to/probe/access_unit.fjob --output /path/to/access.json
```

## Qualification and performance

Every run below uses the pinned v20a model, the four recorded crops and the
frozen policy (SHA256 `381a033d…951f7b`) of the earlier resident qualifications.

The main matrix uses cached instructions and a 20 ms budget: two sessions, two
crops each, one weight upload per session. All 308 layer checks and 20,351,360
layer-output values pass; the maximum local tolerance ratio is .80855, as
before. All 512 head values are bit-identical to the cache-entry, gather and
streamed baselines, verified from independently saved files. Each session then
repeats both crops without intermediate readback and reproduces the same head
bits. All 1,494 main restoration audits pass, the whole arena restores after
both sessions, and the DSP service keeps its PID with zero restarts.

| Frame, cached instructions | Cache entry, 11 modules | Linked, first image | Linked, working copies |
|---|---:|---:|---:|
| Requests, no intermediate readback | about 10,960 | 1,532–1,533 | **52** |
| Requests, per-layer qualification | 10,959–10,964 | 1,566–1,567 | 95–96 |
| Frame body, no intermediate readback | — | 48.8–49.0 s | **1.66–1.69 s** |
| Frame body, per-layer qualification | 153.9–154.6 s | 51.2–51.4 s | 4.3–4.6 s |
| Native wrapper time per frame | about 49.5 s | 30.6–30.8 s | .99 s |
| Maximum wrapper / publication | 7.59 / 8.88 ms | 21.27 / 22.50 ms | 20.33 / 22.06 ms |

The first two columns are the earlier qualifications
([cache entry](RESIDENT_CACHE.md);
[first linked image](../evidence/resident-link-validation.json), revision
`888fbe2`) on a previous boot, not an interleaved benchmark. Those ran at
5 ms and 20 ms budgets respectively; the first linked image at 5 ms needed
5,974 requests.

A frame is now 50 batch requests, one start and one input upload. Native time
by operator class is about .46 s depthwise, .43 s 1×1 convolution and dense
heads, .08 s stem and .03 s pooling and adds. On one crop the successive
changes measured 6.07 s (state and record copies), 2.49 s (inlined gather,
bias window, vector stem), 1.57 s (aligned loads and staged rows), 1.06 s
(call-free unit paths) and .99 s. Of the 1.66 s frame body, roughly .65 s is
host work around the requests, mostly HMX power start (about 5 ms) and close
(1.6 ms) per request; [session-held power](#session-held-hmx-power) removes
that part.

Supplemental runs, each restoring the full arena:

- Load, seal, retire and restore with six metadata rejections, seven batch
  rejections and a competing-owner refusal through the uncached alias, and
  the same fourteen controls through the cached alias inside calibration:
  state, sequence and every arena byte are unchanged after each rejection
  (458 audits for the load run).
- The same image through the uncached instruction alias: identical heads,
  every PC in `0xf1ff0000`, 1,514 requests and 49.9 s. Uncached instruction
  fetch now dominates that control; cached runs report `0x88ff0000`.
- Guarded faults after one HMX transaction and after one in-place depthwise
  unit: native state 4, reuse refused, retire then bounded close, full
  restoration (449 and 462 audits).
- Ordinary SDK direct/benchmark regression afterwards: 72 audits, 38 HMX.
- The access probe: 215 audits.

One development image failed during this work. Its second stride-two load
recomputed the staged row offset, and the per-layer checker rejected
`convdwclip_1`. The session raised `RecoveryRequired` and retained the arena,
refusal record, sealed image and power as designed; nothing was recovered
automatically, and the device was rebooted by its owner before further runs.
The address arithmetic is now checked offline for every depthwise layer. The
final matrix ran on one boot with no failed completion, gate reset, power
recovery or service restart. Sanitized hashes, counts, timings and the probe
table are in
[resident-link-fast-validation.json](../evidence/resident-link-fast-validation.json).
These are audited graph runs on four crops, not sustained tracking latency,
coexistence, thermal or broader accuracy results.

## Session-held HMX power

By default every request powers HMX on under the firmware operation gate and
off again before acknowledging, about 5 ms and 1.6 ms of host time each.
`ResidentLandmark(..., hold_power=True)` instead keeps HMX powered from open to
close:

- **On once.** Open claims the gate, runs the unchanged power-on sequence,
  records the resulting PLL, clock and sequencer registers and the uploaded
  sequencer program, and releases the gate without publishing anything.
- **Proof per request.** Under the gate and before publication, every request
  re-reads those registers. Any difference raises `RecoveryRequired` before
  the DSP is woken; clocks are then left alone, as for any uncertain state.
- **Off last.** Close retires the image and restores the arena as before, then
  claims the gate, repeats the proof and runs the unchanged power-off sequence
  with its restoration check. An open that fails cleanly before ownership
  powers off the same way.

The native images, arithmetic, budget and 40 ms window are unchanged; this is
host-side only. With it a frame takes **1.36–1.39 seconds** for the same 52
requests (1.66–1.69 seconds with per-request power).

Qualification used the same build, crops and frozen policy as above. An idle
hold with no request published showed no register change in 1,193 samples over
three seconds. Calibration and heldout each held power through two qualified
crops, a 30-second idle interval with the gate released (595 and 594 samples,
no change) and two timing frames: all 308 layer checks pass, heads are
bit-identical, 1,495 main audits pass and the power configuration after close
equals the initial one. Guarded faults after one HMX transaction and after one
in-place depthwise unit still invalidate the session, refuse reuse, restore the
arena and power off (449 and 462 audits). Fourteen rejection controls pass
under held power (458 audits). A per-request-power control and the ordinary
SDK regression pass afterwards. Results are in
[resident-power-validation.json](../evidence/resident-power-validation.json).

Holding power is opt-in because it changes what other software on the device
can observe: between requests the gate is free while HMX is on. The per-request
proof detects a changed configuration before this session uses it, but nothing
here establishes how the stock firmware or another client behaves if it wants
HMX power meanwhile, and idle power draw and thermals were not measured. The
default therefore remains per-request power.

```sh
sudo .venv/bin/python tests/hardware_power_hold.py --run-on-device --seconds 3 --output /path/to/hold.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --linked --cached-code --budget-ms 20 --hold-power --hold-idle-s 30 --jobs /path/to/link-jobs --model /private/model --corpus /private/corpus --controls --unqualified-frames 2 --output /path/to/calibration.json
```

## Running

```sh
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --linked --jobs /path/to/link-jobs --model /private/model --corpus /private/corpus --load-only --controls --output /path/to/load.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --linked --cached-code --budget-ms 20 --jobs /path/to/link-jobs --model /private/model --corpus /private/corpus --controls --unqualified-frames 2 --output /path/to/calibration.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --linked --cached-code --budget-ms 20 --jobs /path/to/link-jobs --model /private/model --corpus /private/corpus --split heldout --calibration-report /path/to/calibration.json --output /path/to/heldout.json
```

Heldout requires a completed calibration with the same linked build, alias,
budget, sources, corpus and policy.

```python
from framenpu_cdsp.research.resident_landmark import ResidentLandmark, load_linked

with ResidentLandmark(
    graph, load_linked("link-jobs"), cached_code=True, budget_ms=20, hold_power=True
) as session:
    heads = session.run(chw)
```

The subsequent [depthwise component profile](DEPTHWISE_PROFILE.md) times staging,
gathering, arithmetic, stores and state updates separately, with baseline controls
and numerical validation of each instrumentation variant.

The subsequent [depthwise reuse build](DEPTHWISE_REUSE.md) caches broadcast
coefficients and shares product masks. All four crops retain identical heads;
a sequential pilot reduces depthwise time 7.9% and frame time 6.1%.
