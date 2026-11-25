# Direct dispatch prototype

The opt-in `Device(dispatch="direct")` backend bypasses the stock FLOPS
arithmetic. Its CPU/DSP and HMX paths passed real-device validation on
2026-11-14, including repeated jobs, padded/tiled GEMMs, client ownership and
firmware/power restoration. The default remains `dispatch="benchmark"`
pending sustained tracking coexistence checks.
This is a bounded submission path, not a resident model executor or a general
firmware scheduler.

## Firmware path

On the exact supported CDSP ABI 9 image, the CPU-request loop selects operations
at cached VA `0x870005a0..0x87000600`. A 96-byte replacement routes:

| Operation | Behavior |
|---|---|
| 2 | Existing memory benchmark at `0x870039d0` |
| 4, even or zero callback | Existing FLOPS path at `0x87003850` |
| 4, odd callback | Clear bit 0 and call the custom SDK entry on context 4 |
| 8 | Existing FeatureNN case at `0x87000600` |
| 32 | Existing metadata helper and output cache maintenance |
| Other values | Existing completion path at `0x870006a0` |

The replacement omits the two benchmark-request log calls to fit in the same
window. Standard callback addresses are word-aligned; bit 0 is a private direct
request tag and is cleared before any indirect call. The host publishes the tag
only after checking that this session's replacement remains installed.

Context 4 executes only a scheduling wrapper. It clears the callback tag,
sets worker mode 6 (the callback-only/default worker branch), publishes a
barrier and resumes workers 0/1/2. The usual SDK wrapper executes `frame_job`
on worker 0 and publishes completion from each worker's final return packet.
The master joins all three sequence values, sets worker mode 3 and returns to
stock completion. HMX/HVX instructions remain on worker 0.

Thus no stock FLOPS arithmetic runs for a direct request. The master publishes
state 3 only after all custom workers return. Ordinary benchmark callbacks
continue through the original worker/benchmark code. This mapping has been
checked against the recorded firmware disassembly and the generated replacement.
The direct CPU path, untagged compatibility path and metadata route have also
been exercised on the device; FeatureNN routing has not been load-tested here.

## Installation and removal

Installation is lazy, on the first valid direct submission:

1. Verify packaged source/payload hashes and the installed firmware hash.
   Read the original 96 bytes from that image and check their separate hash.
2. Acquire a process-wide Linux abstract socket lease. A second cooperating
   installer cannot own the patch simultaneously. Reject an already modified
   live window; matching replacement bytes do not establish ownership.
3. Submit the patch helper through the ordinary benchmark callback. Worker 0
   waits up to 20 ms for the master to enter stock completion and for workers
   1/2 to publish their return sequences. It verifies command/owner and all
   expected original words before writing anything.
4. Update the cached execution alias with instruction invalidation, `isync`,
   stores, data clean/invalidate and `syncht`; verify the uncached readback.
   The cache sequence follows the Hexagon self-modifying-code protocol also
   used by [LLVM XRay](https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/xray/xray_hexagon.cpp).
5. Check write status/readback before acknowledging the helper. Partial writes
   or write faults retain the payload and require deliberate recovery.

Clean `Device.close()` submits the same helper through the untagged compatibility
path, replacing the patch with the saved original bytes and checking readback.
Installation and removal each still pay one stock benchmark. Intermediate
submissions reuse the installed firmware edit, without reserving extra code
or tensor storage outside the existing per-job gate.

No firmware file on disk is modified and no proprietary firmware bytes are
bundled. The replacement is built from
[dispatch.S](../src/framenpu_cdsp/sdk/dispatch.S); the helper is
[dispatch_patch.c](../src/framenpu_cdsp/sdk/dispatch_patch.c).

The install/remove job is the one job during which the firmware's main thread sits in its own timed
wait (state 3, a 50 ms deadline held in `r1:0`) while the job's vector table is in EVB, which every
hardware thread shares. The table's forwarding stubs must therefore change no register: up to 0.1.0
they loaded their target into `r0`, so an interrupt of the main thread could expire the deadline at
once and leave the DSP at `[1,1,1]`. `tests/test_vector_stubs.py` checks the shipped payloads.

## Compatibility and failures

The SDK entry now supports the context-4 scheduling wrapper. Newly compiled
FJOBs declare `supports_direct=true`; old files without that optional metadata
remain usable on the benchmark backend and must be rebuilt for direct mode.
The model format and public `submit` arguments are unchanged. In both modes,
`JobResult.workers` still contains the three worker status rows.

The 40 ms host timeout remains. A stuck direct job keeps the master waiting
for worker returns; timeout does not authorize overwriting its code/data or
switching HMX off. Once the master publishes completion, the stock roughly
50 ms acknowledgment window still applies. Direct mode adds no arbitrary
native-job cancellation, crash-proof host cleanup, or cross-client isolation.

If the transport is poisoned, `close()` leaves the firmware edit and potentially
active payload in place and closes only host resources. A later process rejects
the modified window; the abstract lease disappearing after a process crash is
not evidence that restoration is safe. Recovery must establish DSP idle/reset
and the original firmware state. No automatic service restart is performed.

## Measurement and device validation

`JobResult.elapsed_ms` measures publication through observed completion;
uploads, power setup/teardown and cleanup are outside it. Wrap the whole call
for end-to-end timing. `kernel_reference_ticks` reports worker-0 `frame_job`
time from raw `c31:30` counters in status words 10..13; it excludes entry
VTCM backup and exit restoration. Old status rows without timers return `None`.

The host permits the brief interval in which all workers have returned while
the master still owns processing state 2. It waits for state 3; an expired
ownership/completion state still requires recovery.

### Observed CPU results — 2026-11-14

There were 100 warm samples per probe/backend, after one first call. These are
wall times around the complete `submit` call, including upload and cleanup:

| Probe | Benchmark p50 / p99 | Direct p50 / p99 |
|---|---:|---:|
| Empty job | 4.294 / 4.452 ms | 0.734 / 0.898 ms |
| Scalar 4 KiB byte copy | 12.634 / 12.932 ms | 9.127 / 9.395 ms |

The empty-job median improved about 5.85x; the copy probe saved about 3.51 ms.
The byte-copy kernel itself remained around 8.34 ms, so these results identify
dispatch savings and a remaining scalar-transfer cost. They do not establish
optimized copy bandwidth or HMX throughput. The first direct empty call,
including installation, took 13.78 ms.

Custom C vector addition, ReLU, clipping, broadcast addition and recovery from
an unsupported opcode passed. A second process's direct installer was refused;
a legacy client still ran with the patch installed and the original owner
remained usable. The stock metadata name hash matched before/during the patch.
Afterward, the original 96-byte window and the first 112 KiB of firmware code
matched the installed image; callback was zero, control was `(1,0,1)` and the
DSP service retained its PID with zero restarts. These checks do not establish
eye-tracking quality. See the sanitized
[numerical summary](../evidence/dispatch-validation.json).

### Observed HMX results — 2026-11-14

Both backends passed the five padded/tiled GEMMs through `(33,65,35)`, DSP
activations and a serialized/reloaded XOR model. The deterministic inputs
matched the independent tile-precision reference exactly. The qualification
harness audited restoration before acknowledgment for 78 correctness jobs,
including 39 HMX jobs: all borrowed DDR spans, callback and touched HMX power
configuration matched their values from gate acquisition.

A separate comparison used 100 warm samples per backend. Timings below are
for one 32x32 identity GEMM with checked output; the restoration audits were
disabled during timing to avoid inflating the measurements.

| Measurement | Benchmark p50 / p99 | Direct p50 / p99 |
|---|---:|---:|
| Complete `submit` wall time | 29.340 / 39.095 ms | 25.359 / 37.309 ms |
| Publication to observed completion | 19.890 / 20.289 ms | 16.010 / 16.325 ms |

The complete-call median improved by about 3.98 ms (13.6%). These are still
per-job uploads, power cycles, scalar packing and output work. The reference
counter covers `frame_job`, including packing and readback; it does not isolate
HMX arithmetic. Bounded multi-tile execution and HVX packing remain necessary
before evaluating useful convolution throughput.

Final worker SSR values were `0x81840200`, `0x89840200`, `0x91840200`.
Power configuration, the dispatcher window and the first 112 KiB of firmware
code were restored; control was `(1,0,1)`, callback zero, and the service kept
its PID with zero restarts. No sustained gaze/thermal result is implied. See
the sanitized [HMX qualification summary](../evidence/hmx-dispatch-validation.json).

The opt-in comparison script checks empty jobs and exact 4 KiB copies, separates
the first call from warm p50/p95/p99 timings, checks an untagged submission with
the patch active, and reconnects to verify restoration after close:

~~~sh
python tests/hardware_dispatch.py --run-on-device --dispatch benchmark --iterations 1
python tests/hardware_dispatch.py --run-on-device --dispatch direct --iterations 1
python tests/hardware_dispatch.py --run-on-device --iterations 20
sudo .venv/bin/python tests/hardware_dispatch.py --run-on-device --iterations 20 --hmx
sudo .venv/bin/python tests/hardware_smoke.py --run-on-device --dispatch direct
sudo .venv/bin/python tests/hardware_hmx.py --run-on-device --iterations 100
~~~

The optional GEMM case is a repeated 32x32 identity product with a checked
reference. It still uploads operands and scopes HMX power per job; it is not
a resident-weight throughput measurement. Hardware qualification must also
observe tracking coexistence. The combined `hardware_hmx.py` harness first
audits correctness/restoration, then runs the separate timing comparison and
checks final worker, clock and firmware state.
