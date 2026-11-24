# Repeated resident GEMM proof

The bounded resident prototype loads weights once and streams different inputs
through true **32×K×32 GEMMs**, K=32/64/128. On 2026-11-15, 300 measured jobs,
12 checked warmups and six additional reuse jobs matched all 325,632 FP32 output
values. Twenty-one invalid commands were rejected. Three guarded cached traps
were cleaned up and followed by clean teardown and a new session with different
weights and a different instruction image. All 372 resident restoration audits
passed, with the service retaining its PID and zero restarts.

This implements the fixed-layout proof from the [resident design](RESIDENT.md).
It does not execute arbitrary planner schedules. The general graph loader,
real convolutions, model accuracy and sustained tracking remain future work.
The [sanitized evidence](../evidence/resident-gemm-validation.json) includes all
command timings, input/reference hashes, source and payload hashes, rejected
commands, fault checks and normal-SDK regressions.

## Storage and execution

[ResidentGemm](../src/framenpu_cdsp/research/resident_gemm.py) acquires an exclusive Linux
abstract socket lease before opening its direct-dispatch Device. Its 32 KiB
reservation is reproduced by the offline planner:

| DDR offset | Bytes | Purpose |
|---|---:|---|
| `0x1a22000` | 4,096 | Stored code image; checked for unexpected changes |
| `0x1a23000` | 4,096 | Host ownership record and uncached native session state |
| `0x1a24000` | 8,192 | Weights, prepacked once by DSP |
| `0x1a26000` | 8,192 | Changing tile-major FP16 inputs |
| `0x1a28000` | 4,096 | Full row-major FP32 output |
| `0x1a29000` | 4,096 | Boundary guard, unchanged after every command |

Only the active K-dependent portion of the input/weight slots is used. Every
depth tile has distinct addresses and data; there is no repeated-block reduction
or operand aliasing. Weights remain in DDR across requests, with their full
8 KiB slot hash checked after loading and every following command. Only inputs
are uploaded for ordinary runs. Each session records exactly one weight upload.

The stored image is copied into the existing borrowed 4 KiB executable slot
by the ordinary submission path. Execution uses its experimentally established
cached alias. The proof does not execute resident DDR in place or expand VTCM:
HMX and HVX stay on worker 0 within the existing backed-up/restored 8 KiB.
All participating workers return before completion. HMX power is started and
restored for every command, including load/cleanup commands; idle residency
does not keep power or the firmware operation gate reserved.

The kernel packs weights in resident DDR once on load. For each run it copies
each distinct A/B depth tile to VTCM, packs A, clears HMX once, loads all depth
tiles, and stores once. The final FP16 HMX result is promoted to FP32 with the
existing exact HVX conversion. This arithmetic differs from the public API's
FP16 depth-partials summed in FP32. There are no biases, tails or activations.
The corpus uses signed integers in eighths and an independent integer-derived
full-K reference; these results do not choose production numerical tolerances.

## Protocol and resource lifetime

The native [probe](../tests/resident_gemm.c) accepts a fixed 128-byte descriptor
inside ABI 1's temporary job input and returns 256 bytes of metadata there.
It checks the existing header/arena bounds and direct/HMX flags before using
VTCM. Descriptor words are:

| Words | Meaning |
|---|---|
| 0..3 | Magic `0x5247454d`, protocol 1, command, depth tiles 1..4 |
| 4..7 | Nonzero 128-bit session nonce |
| 8..9 | Nonzero 64-bit command sequence |
| 10..17 | SHA-256 identity of code, K and serialized weights |
| 18..23 | Exact weight/input/output offsets and active lengths |
| 24 | Compiled image tag, 0 or 1, checked by its corresponding image |
| 25..31 | Reserved zero |

Commands are initial cache reconciliation (0), load/prepack (1), run (2),
qualification trap (3), and close (4). Load initializes native identity/state;
following commands must match it and advance the last accepted sequence by
exactly one. Invalid commands return ABI error 1 without advancing the native
sequence or modifying the output. The host reserves the final sequence value
for close, before wrap. Native code compares the opaque content identity; the
host checks stored code bytes and packed weight bytes/hashes. Reconciliation
command 0 is an initialization operation without loaded native state.
The host checks command echo, nonce/content/sequence, cached
PC, worker identities, errors, SSR/EVB and cleanup markers before reclamation.
The tag-1 replacement image rejects a tag-0 descriptor and vice versa, allowing
the fresh-session test to detect stale instruction execution.

Initialization writes the uncached ownership marker first, then the nonce,
content identity and native state. It runs inside the first real submission's
prepare hook while the firmware gate is held. Weights and inputs are likewise
uploaded through that hook. A second cooperating owner is refused by the socket
lease; if that socket disappears after a crash, the DDR marker still refuses
adoption. Matching data or a fresh host process does not authorize clearing it.
Host tests exercise this crash/refusal boundary and timeout resource retention.

The [entry](../tests/entry_resident.S) derives from the qualified cached entry.
It reconstructs FP when abandoning C frames, cleans/invalidates the complete
24 KiB resident data window and 4 KiB cached code image, and confirms status
markers `(3,3)` before normal or guarded-fault completion. Control and the stored
image are outside the cached data window. This conservative cleanup gives the
host a clean boundary before writing new inputs or reading outputs.

The fault control dirties cached output with a sentinel and traps with cause 37.
The host requires exception code 2, the cached ELR, flushed sentinel and no C
continuation. The session becomes invalid and refuses more work. Close remains
available because all workers and cache cleanup have been validated. A timeout,
bad completion, changed weight/guard or unconfirmed cleanup instead retains code,
arena and power and requires deliberate recovery. No automatic restart, gate
clearing, replay or recovery is implemented.

Clean close reconciles caches, validates native retirement and restores the
owned code/data/state while the completed close request still holds the gate.
It then removes the direct dispatcher and retires the host-only ownership prefix
last, under the arena lease. All 32 KiB must match the pre-session snapshot.
If any step fails, the refusal marker remains or is reestablished.

Preliminary controls exposed stock processing state 2 after a claim/release
without publication. Their GEMM values and borrowed-memory audits passed, but
the strict post-close health check failed. The final implementation removed
these claim-only windows: initialization and restoration occur within real
submissions. A bounded, read-only idle observer never resets a gate or retries
an active request, callback, poison, power mismatch or changed firmware prefix.
The preliminary controls are excluded from qualification counts.

## Measurements

These are audited qualification timings with 100 measured runs per K; four
numerically checked warmups per point are excluded from percentiles. References
are calculated before submission. Times are observed on the supported firmware
with the existing power policy and unchanged 40 ms host limit.

| Phase, median | K=32 | K=64 | K=128 |
|---|---:|---:|---:|
| Resident input upload, host | 0.99 µs | 1.04 µs | 1.15 µs |
| DSP operand copy to VTCM | 5.52 µs | 10.42 µs | 20.21 µs |
| DSP A packing | 0.052 µs | 0.104 µs | 0.677 µs |
| DSP HMX clear/load/store intervals | 3.49 µs | 2.92 µs | 2.92 µs |
| DSP output conversion/stores | 8.91 µs | 8.91 µs | 8.96 µs |
| Cached code publication / entry work | 0.823 ms | 0.824 ms | 0.825 ms |
| Data/code cleanup and C return | 3.592 ms | 3.595 ms | 3.588 ms |
| HMX power start, host | 4.825 ms | 4.716 ms | 4.634 ms |
| HMX power close, host | 1.489 ms | 1.480 ms | 1.480 ms |
| Complete command p50 | 12.578 ms | 12.509 ms | 12.404 ms |
| Complete command p99 | 17.774 ms | 22.102 ms | 23.548 ms |

The reference timer's resolution is 52.08 ns; no timer-overhead subtraction is
applied. Short HMX/packing intervals are descriptive, not a throughput slope.
`resident_upload_ms` is a subinterval of `prepare_ms`; phase percentiles are
not additive. DSP cleanup ends before the existing VTCM/register restoration.
The command timer excludes input validation/FP16 tile-major encoding and CPU
reference construction; input encoding is recorded separately per run. Weight
serialization happens before session open; DSP weight packing is timed on load.
Restoration audit costs are included in the host claim/ack/validation intervals.

Session open costs 43.6–51.9 ms, including dispatch installation, reconciliation
and load; close costs 16.1–17.3 ms. Those are multiple bounded requests, not one
callback exceeding its deadline. Maximum measured wrapper span is 4.613 ms and
publication-to-completion is 5.674 ms. All 372 audits pass, including 360 HMX
commands and 12 direct-dispatch helpers. Six sessions restore their full owned
span. Gate/callback, firmware prefix and power snapshots match before/after.

The existing public GEMM/activation/custom-job/saved-XOR suite subsequently
passes 78 more restoration audits, including 39 HMX. Its one-sample timing is
smoke coverage only. Combined resident and ordinary regression audits are 450;
preliminary controls and the small fault pilot are outside that count.

Full-window cache cleanup and per-request power costs dominate. Range-specific
cache maintenance and any longer power ownership need further proof. This is
not full-model throughput or a real-time tracking result.

## Reproduction

Build both distinct instruction images with LLVM 23.1.4:

~~~sh
python3 handoff-tools/build_mw.py tests/entry_resident.S tests/resident_gemm.c \
    build/resident.fjob build/resident
python3 handoff-tools/build_mw.py tests/entry_resident.S tests/resident_gemm_v2.c \
    build/resident-v2.fjob build/resident-v2
~~~

Both payloads are 3,668 bytes; their distinct hashes and source hashes are in
the evidence. The experimental Python interface accepts a qualified image and
an ordinary K×32 weight matrix:

~~~python
from framenpu_cdsp import Job
from framenpu_cdsp.research.resident_gemm import ResidentGemm

with ResidentGemm(Job.load("build/resident.fjob"), weights) as gemm:
    for inputs in input_stream:  # each input is 32×K
        output = gemm.run(inputs)
~~~

After checking idle firmware/power state, run a small control,
then the guarded suite on the supported Frame:

~~~sh
sudo python3 tests/hardware_resident.py --run-on-device --probe build/resident.fjob \
    --phase controls --sizes 32 --iterations 1 --warmups 0 --output /tmp/resident-control.json
sudo python3 tests/hardware_resident.py --run-on-device --probe build/resident.fjob \
    --replacement build/resident-v2.fjob --phase suite --sizes 32 \
    --iterations 1 --warmups 0 --output /tmp/resident-pilot.json
# After both pass:
sudo python3 tests/hardware_resident.py --run-on-device --probe build/resident.fjob \
    --replacement build/resident-v2.fjob --phase suite --sizes 32 64 128 \
    --iterations 100 --warmups 4 --output /tmp/resident-full.json
~~~

The later [convolution extension](CONVOLUTION.md) adds a separately qualified
tag-2 image with native FP32 bias/clip. The original timing/evidence hashes above
remain those of this GEMM comparison.

The native jobs remain trusted supervisor code. A guarded software trap does
not establish arbitrary corruption or pipeline-stall recovery. The next model
milestone is a real convolution against pinned ncnn references, with explicit
packed-weight and numerical tolerances.
