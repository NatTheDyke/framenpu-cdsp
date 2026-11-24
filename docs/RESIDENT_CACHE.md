# Cached execution for the resident graph

`ResidentLandmark(..., cached_code=True)` opts into a resident-specific cache
entry. Kernel instructions execute through the cached alias of the existing
4 KiB code slot. Weights, FP32 tensors, control state and scratch data retain
their uncached addresses. The default remains `cached_code=False`.

This step changes instruction execution and its publication/cleanup protocol.
It preserves numerical policy, arithmetic order, tensor layout, one HMX owner,
32-unit compute/256-unit pack/post caps, the five-millisecond between-unit
budget, 40 ms host timeout, 8 KiB VTCM backup and 128 KiB borrowed arena.
It does not fuse operators or enlarge graph segments.

## Entry and compiler contract

Build the modules with the versioned entry:

```sh
python tools/build_resident_landmark.py --cached-entry --output /path/to/cache-jobs
python tools/check_resident_cache.py --jobs /path/to/cache-jobs --output /path/to/entry-verification.json
```

The replacement/fault controls use two small images built with the same entry:

```sh
python handoff-tools/build_mw.py kernels/resident_graph_entry.S tests/resident_cache_probe.c /path/to/probe-a.fjob /path/to/build-a
python handoff-tools/build_mw.py kernels/resident_graph_entry.S tests/resident_cache_probe_v2.c /path/to/probe-b.fjob /path/to/build-b
```

Each FJOB carries the `resident_graph_KIND_cache_v1` name. The controller checks
that contract before opening a device when cached execution is selected. The
same module bytes support both cached and uncached control modes, selected by
context parameter zero. Other context parameters remain reserved and zero.
The selected mode is bound into the session identity.

The entry first installs the guarded exception window and saves the existing
8 KiB VTCM span. It stages the entire freshly uploaded 4 KiB code image in the
first 4 KiB of that saved scratch. Before any cached stores, it arms worker-0
status word 14. Full-vector cached writes precede data-cache clean/invalidation,
then instruction-cache invalidation. The entry tail-jumps to the cached
`frame_job` address while preserving an uncached return address.

The normal and guarded synchronous-fault paths return through uncached assembly.
They clean/invalidate the full code window and set status word 15 only after
completion. VTCM, SSR/EVB and entry frames then restore before worker publication.
Every host completion checks both cleanup markers and the selected instruction
PC before code/power reclamation. Fault ELR must also belong to the selected
alias. Missing confirmation retains code, power, arena and refusal record.
There is no automatic reset or recovery for a stall or cleanup failure.

To fit within the existing reservation, exception-forwarding stubs now share a
small trampoline. All sixty forwarded vectors still reach the original firmware
target with the same `r0` value and no extra register clobbers. Vector spacing,
256-byte alignment, four guarded vectors and the 80-byte entry frame remain
unchanged. The verifier checks linked tables and cached tail targets for every
module.

Cached builds disable compiler jump tables and reject non-PC-relative C code
relocations. An initial diagnostic calibration passed numerically, but static
inspection found the admin switch table contained uncached branch targets.
The final build removes that table and repeats calibration/heldout qualification.
The other ten modules retain identical code bytes through that correction.
Source, toolchain flags, code hashes and relocation results are recorded with
the build. Public SDK entry/runtime payloads remain unchanged.

## Qualification and performance

The small entry controls alternate two distinct instruction images and both
aliases, checking exact XOR output, code tags, PCs and cleanup markers. Sixteen
guarded traps are followed by successful normal reuse. Invalid parameters return
without output writes. These controls pass 53 restoration audits. Separate
load/rejection/ownership/whole-arena teardown controls pass in both modes.

The copy fixture's median body time falls from 38.67 to 4.92 microseconds;
its complete wrapper grows from .073 to 1.608 milliseconds because code
publication/cleanup is paid on each request. These measurements establish the
entry contract and show why small requests still need amortization.

The final verified build passes all four crops under the same pinned v20a
model, local arithmetic rules and frozen propagated maximum/RMS policy as
[the gather baseline](DEPTHWISE_GATHER.md). All 308 layer checks, 360 blob checks
and 20,351,360 layer-output values pass. Independently saved outputs contain
512 values, all bit-identical to the gather and earlier streamed/resident
baselines. Two sessions each reuse one weight upload across two different
inputs. All 44,739 main restoration audits pass, with independent whole-arena
restoration and unchanged firmware, power configuration and DSP service.
Every command confirms its cached PC and both cleanup markers.

| Four-crop audited measurement | Uncached gather baseline | Cached entry |
|---|---:|---:|
| Frame body range | 183.6–187.1 s | 153.9–154.6 s |
| Total frame body time | 743.3 s | 616.9 s |
| Requests per frame | 14,217–14,233 | 10,959–10,964 |
| Depthwise native wrapper total | 167.2 s | 136.8 s |

The historical four-crop comparison improves total audited frame time **1.20×**,
a **17.0% reduction**. Faster execution lets some native loops finish more work
before the unchanged five-millisecond between-unit budget; the unit caps stay
fixed. Maximum cached wrapper/publication intervals are 7.586/8.877 ms. Planning,
load, transfer, qualification, per-request power, execution and close timings
remain separate in the evidence. These are audited graph runs, not sustained
production latency or a statistical alternating model benchmark.

A sequential same-payload control runs the first calibration crop with caching
disabled: 196.95 seconds and 14,414 requests versus 153.88 seconds and 10,962
requests cached (1.28×). All work-unit totals and output bits match. Both modes
pass native rejection, exclusive-owner and full-arena teardown controls. This
single control isolates the selected instruction alias within identical module
bytes; the four-crop historical comparison above is a different measurement.

Native cached faults after one HMX transaction and one in-place depthwise unit
invalidate the session and refuse reuse. Both allow bounded close and restore
the full arena, passing 465 and 4,558 audits. Ordinary SDK direct/benchmark
regression passes another 72 audits. No failed completion, reboot, gate reset,
power recovery or service restart occurs during this qualification. Sanitized
hashes, checks and timings are in
[resident-cache-validation.json](../evidence/resident-cache-validation.json).

## Running and reproducing

With private references prepared as in [LANDMARK.md](LANDMARK.md), run explicit
hardware qualification on the supported Frame as root:

```sh
sudo .venv/bin/python tests/hardware_resident_cache.py --run-on-device --probe-a /path/to/probe-a.fjob --probe-b /path/to/probe-b.fjob --output /path/to/cache-controls.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --jobs /path/to/cache-jobs --model /private/model --corpus /private/corpus --cached-code --controls --output /path/to/calibration.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --jobs /path/to/cache-jobs --model /private/model --corpus /private/corpus --cached-code --split heldout --calibration-report /path/to/calibration.json --output /path/to/heldout.json
```

Heldout qualification requires a completed calibration with the same instruction
mode as well as matching module, source, corpus and policy hashes. Omitting
`--cached-code` runs the same cache-entry module bytes through their uncached
control path. Original ordinary-SDK modules remain supported by the default
uncached controller; selecting cached execution with those jobs is rejected
before device access.

```python
from framenpu_cdsp.research.resident_landmark import ResidentLandmark

with ResidentLandmark(graph, cache_jobs, cached_code=True) as session:
    first = session.run(first_chw)
    second = session.run(second_chw)
```

Weights still upload once per session and all intermediate tensors remain
resident. Cached instructions alone do not establish real-time tracking,
broader accuracy, sustained coexistence or thermal qualification. Model weights,
recordings and raw device dumps remain private.
