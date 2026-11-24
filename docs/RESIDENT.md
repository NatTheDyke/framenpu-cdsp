# Resident executor: storage and session design

Step 1 implements an **offline storage planner** and defines the contract for
the next executor prototype. `framenpu plan-resident` reads a declarative JSON
schedule, validates storage dependencies, and emits a bounded allocation map.
It opens no device; its generic schedule output sets `execution_supported=false`.
Step 2 now implements a separate [bounded resident GEMM proof](RESIDENT_GEMM.md)
with fixed descriptors, weights loaded once, changing inputs and qualified
cache/fault teardown. Step 3 adds [three real-weight convolution layers](CONVOLUTION.md)
with frozen ncnn numerical comparisons. The separate
[pinned resident landmark executor](RESIDENT_LANDMARK.md) implements its fixed
v20a graph with persistent weights and FP32 activations. General graph loading
and execution of the generic planner's schedules remain future work.

The design targets the exact firmware hash in [profile.py](../src/framenpu_cdsp/profile.py).
Its evidence is the [memory map](MEMMAP.md), [direct dispatcher](DISPATCH.md),
[cached-code experiment](HMX_CACHE.md) and [fault-unwind qualification](FAULT_UNWIND.md).
The fixed proof's new hardware qualification is documented separately.

## Memory and ownership

All offsets below are relative to the 32 MiB shared DDR mapping. Plan output
uses these offsets, rather than host virtual pointers. The future executor
must validate offset plus length before deriving either DSP alias.

| Resource | Location / budget | Lifetime and ownership |
|---|---|---|
| Resident backing storage | DDR `[0x1a22000,0x2000000)`, 6,152,192 bytes | Exclusive cooperating session; firmware-profile checks and arena lease required before writes |
| Code image | `@code` within resident DDR; default 4 KiB | Immutable stored image for the session; this reservation does not establish execution permission or relocation correctness |
| Control/descriptors | `@control` within resident DDR; default 4 KiB | Session-long budget; exact wire encoding must be sized and versioned before loading |
| Packed weights/biases | Named `weight` allocations in resident DDR | Immutable, uploaded once, live throughout all runs |
| Inputs/intermediates/outputs | Named allocations in resident DDR | Reused according to each bounded run's ordered schedule and readback boundary |
| Packing/FP32 carry/double buffers | Per-step `scratch_bytes` in resident DDR | All simultaneous buffers must be included in this byte budget |
| Bootstrap executable | Existing DDR `[0xa92000,0xa93000)` | Borrowed under the stock operation gate, restored before acknowledgment |
| Job header/status/temporary arena | Existing SDK reservations | Borrowed per submission; never used as persistent model storage |
| HMX VTCM scratch | Existing first 8 KiB only | Borrowed, backed up and restored inside each operation-gate window |
| Workers/HMX/power | Current dispatcher and power protocol | One complete HMX transaction owner; temporary ownership ends only after validated return and restoration |

DDR is the backing store. The plan assigns no new VTCM addresses and preserves
FeatureNN/gaze reservations. A stored code image can initially be copied into
the existing 4 KiB executable reservation for bounded work. Execute-in-place
from resident DDR, larger modules and relocation need separate proof. A code
image reservation is capped at 4 KiB in this planner revision.

The future resident lease must cover the entire arena across runs. It is
separate from the per-request firmware operation gate and must cooperate with
the existing direct-dispatch lease. All resident writers must share the same
lease implementation before any of them is enabled. Matching bytes or a
vanished process/socket do not authorize attaching to an abandoned arena.
A host crash loses the ownership proof; reconnect requires established idle or
reset state and cache reconciliation before storage can be reused. The lease
coordinates participating clients; it cannot isolate trusted native clients.
Acquire locks in arena-lease, direct-dispatch-lease, firmware-gate order and
release in reverse order. A healthy idle session retains the arena lease;
the firmware gate is reacquired for each bounded request. RecoveryRequired must
leave a persistent ownership/poison record for a future cooperating process;
process-local lock disappearance must not erase that record.

The initial executor keeps borrowed VTCM and power scoped to each bounded
segment. Persistent weights do not justify holding the firmware gate or power
configuration across idle intervals. Any later power/session amortization
needs its own coexistence and ownership qualification. The linked landmark
executor has since gained an opt-in, ownership-qualified
[session-held power mode](RESIDENT_LINK.md#session-held-hmx-power); coexistence
with another HMX power user remains unestablished.

## Offline schedule and allocation

~~~sh
framenpu plan-resident examples/resident_residual.json
~~~

Python callers can use
`framenpu_cdsp.resident.plan_resident(spec)` or `load_resident_plan(path)`.
The standalone schedule does not import FNPU/ncnn graphs or alter
`Model.plan()` or `Model.run()`.

Version 1 requires `format="framenpu-resident-plan"`, integer `version=1`,
`name`, `tensors`, ordered `steps` and final `outputs`. Optional `arena_bytes`
can lower the memory budget; it cannot enlarge the verified span.
`code_bytes` defaults to 4096 and `control_bytes` to 4096. Both consume space
in addition to the tensor and scratch budgets. Unknown fields are rejected.

Each tensor requires:

- `name`, unique within the tensor table, and `kind`: `input`, `weight` or
  `activation`.
- `shape`, one to eight positive integer dimensions, and `dtype`: `u8`, `f16`
  or `f32`.
- `layout`, an explicit storage label such as `nhwc`, `matrix` or `linear`.
  The planner carries this label; a future kernel must validate its semantics.
- Optional byte `strides`; defaults are contiguous row-major strides. Explicit
  strides must be positive, dtype-aligned and nonoverlapping in row-major order.
  The charged span includes padding through the last addressed element.
- Optional `alignment`, a power of two from 128 to 4096 bytes, default 128.
- Optional `alias_of`, for whole-buffer activation views of the same dtype.
  Contiguous reshapes and identical padded views are supported. Aliases reserve
  no new storage and inherit their root's lifetime. Partial slices, transposes,
  dtype reinterpretation and alias writes are not supported.

Each step has a unique `name`, lists of `inputs` and fresh activation `outputs`,
and optional `scratch_bytes`. The caller must account for actual packed weight
formats, bias storage, FP32 accumulators and every simultaneous scratch/double
buffer. The planner checks byte storage and scheduling; it cannot infer missing
kernel requirements, shapes or numerical semantics. Descriptor tables that
exceed the control reserve must be rejected by the future loader.
FP32 carry or a packing buffer consumed by a later step must be a named
activation with the corresponding dependency, rather than one-step scratch.
This revision schedules steps serially; concurrent/double-buffered execution
needs overlapping lifetimes represented explicitly before it can use the map.

Time is represented by half-open intervals `[start,stop)`. Setup is event 0,
operations are events 1 through N, and final readback is N+1. Inputs begin at
setup; activation roots begin at their producer. Every read extends the root
through the consuming event. Inputs and outputs of the same step coexist, so
storage becomes reusable only at a later event. Final outputs, including alias
roots, remain live through readback. Weights, code and control stay live over
the entire schedule. A new run must restore inputs and produce all activations;
only immutable weights/code and control are retained between runs.

Placement is deterministic first-fit in producer order, with session-long
objects placed before transient storage. Alignment and sizes rounded to
128-byte boundaries prevent simultaneously live roots from sharing cache lines.
Alias alignment requirements propagate to the root. `required_bytes` is the
placement high-water mark; `peak_live_bytes` sums simultaneous rounded spans;
their difference is placement/alignment overhead. `without_reuse_bytes` places
the same requests consecutively with their alignments. The allocator is greedy:
failure means this placement/schedule exceeds the budget, not that every
possible rearrangement is impossible.

Names use ASCII letters/digits/underscore/dot/hyphen, start with a letter or
underscore, and are at most 64 characters. JSON input is limited to 2 MiB and
1024 tensors/steps, with duplicate fields and duplicate names rejected. Inputs
and weights are available at setup; every activation root has exactly one
producer. Reads before production, in-place writes and alias cycles fail
preflight. Plan output is a report: a future loader must revalidate its source
schedule rather than trust edited output offsets or its firmware hash field.

### Residual example allocation

[resident_residual.json](../examples/resident_residual.json) describes two
pointwise-sized storage steps, residual addition and clipping. It is a synthetic
storage example; its weights, packing budget and operator names do not establish
an implemented convolution. Byte offsets below are relative to arena start
`0x1a22000`; the CLI emits offsets relative to the full DDR mapping.

| Allocation | Arena-relative offset | Reserved bytes | Lifetime |
|---|---:|---:|---|
| Code / control | 0 / 4,096 | 4,096 each | `[0,6)` |
| Weight 0 / weight 1 | 8,192 / 9,728 | 1,536 each | `[0,6)` |
| Input / skip input | 11,264 | 49,152 | `[0,4)` |
| Hidden | 60,416 | 65,536 | `[1,3)` |
| First-step scratch | 125,952 | 12,288 | `[1,2)` |
| Projected | 125,952 | 49,152 | `[2,4)` |
| Second-step scratch | 175,104 | 12,288 | `[2,3)` |
| Residual sum | 60,416 | 49,152 | `[3,5)` |
| Clipped / flat output alias | 11,264 | 49,152 | `[4,6)` |

Required storage and peak live storage are both **187,392 bytes** (183 KiB).
Without reuse the same requests take 297,984 bytes (291 KiB). The input remains
live until the residual addition; only the later clip may reuse its space.
The flat output aliases the clipped tensor without another allocation.

## Proposed session and completion contract

The following is an implementation contract for step 2, not an extension to the
current ABI 1 header. Keep the existing job path for comparison and bootstrap.

| State | Permitted next action |
|---|---|
| New | Validate firmware, schedule, packed data sizes/hashes and storage bounds; acquire exclusive arena ownership and inspect idle state |
| Loading | With execution stopped, initialize the generation/control block, upload images/weights, reconcile caches and validate readback before marking ready |
| Ready | Upload inputs and publish one bounded segment while holding the firmware operation gate |
| In flight | Host leaves payload, code, control and power intact until all participating workers return |
| Returned | Validate identity, descriptors, worker status and cache/cleanup completion before exposing output or releasing temporary resources |
| Ready after success | Continue a validated dependent segment or read final output; intermediate tensors remain on DSP |
| Invalid after a confirmed fault | Mutable tensors/outputs are unusable; only validated cleanup and teardown are allowed before a fresh session |
| Recovery required | Retain possibly active resources and stop submitting; deliberate recovery must prove idle/reset and restore state |
| Closed | After validated idle, reconcile/invalidate caches, restore borrowed state, retire the generation and release ownership |

Use a new host-generated 128-bit session nonce plus a monotonic 64-bit command
sequence. Bind each command and completion to that nonce, sequence, plan/content
identity and segment index; reject zero or reused identities and rotate the
session before sequence wrap. The bootstrap still verifies ABI 1's per-job
sequence and all worker returns. The executor must check the resident identity
in addition, so a completion from a prior session/reset cannot validate a new
command. The nonce detects stale records; it does not establish firmware idle.
Exact descriptor encoding, status size and native validation remain step-2 work.

All descriptor ranges use checked offset/length arithmetic against owned spans.
The loader must enforce read-only weight/code storage, declared layout/dtype,
tensor alias rules and each segment's input/output/scratch ranges. Segment
completion is published after required cache maintenance and all worker joins.
Validate completion before host reclamation/power-down, as in the existing
`_completion_validator` boundary. A malformed completion or uncertain publication
is RecoveryRequired, including when the stock firmware gate has expired.

Keep the existing 40 ms host limit and roughly 50 ms firmware acknowledgment
window. Residency allows data to survive separate bounded requests; it does
not allow a long-running callback or authorize extending either deadline.
Cancellation can be checked at a completed tile/segment boundary, followed by
all-worker return and validated cleanup. Timeout, host disconnect or cancellation
intent alone cannot authorize reuse. A guarded trap qualifies the public entry's
unwind; every new resident/cache path still needs its own fault/cleanup proof.

## Cache handoff rules

The host maps DDR uncached, while DSP code/tensors may use cached aliases.
Track a single writer and explicit ownership for each cache-line-rounded root.
Alias descriptors share that ownership. Never mix host/uncached writes with
dirty DSP aliases or overwrite an image that can still execute.

1. Before a host upload, finish prior DSP work and reconcile any cached lines
   that cover the span. Host upload completes before descriptor publication.
2. Before DSP cached reads, establish the uploaded bytes at the cached alias
   with the validated data publication/invalidation sequence and barriers.
   Invalidation instructions that also clean cannot discard stale dirty data.
3. Keep intermediates DSP-owned across segments only with defined visibility
   between producers and consumers. A CPU read or another writer requires
   explicit cache reconciliation; DDR residency alone supplies no coherency.
4. Before host readback or teardown, clean/invalidate DSP-written spans and
   synchronize before publishing completed status. No remaining cached writer
   may later overwrite a reused line.
5. Publish code only with execution stopped. Reuse the experimentally qualified
   staged full-line publication and instruction synchronization where applicable;
   changed addresses, modules or execution lifetimes require new qualification.
   Fault cleanup must run from code whose validity does not depend on the image
   being replaced, and its completion must be checked before reclamation.

These are required transitions. The planner implements none of the cache
operations and chooses no new instruction sequences. Step 2 must demonstrate
host write → DSP cached read, DSP write → host read, different-input reuse,
code replacement, stale completion rejection and guarded-fault cleanup on the
actual resident path before reporting its performance.

## Hardware milestone and next work

The [bounded GEMM proof](RESIDENT_GEMM.md) now qualifies distinct weight/input
depth tiles, session reuse, stale identity/bounds rejection, cached dirty-fault
cleanup, code replacement and full-span teardown. Its native descriptor encoding
implements the fixed subset of this design; arbitrary planner output is still
an offline report. Three real convolutions now pass [pinned ncnn comparisons](CONVOLUTION.md)
with frozen arithmetic/total-error rules. Next reduce spatial-request overhead
and extend the graph while checking propagated numerical error.
Sustained gaze quality still needs its separate tracking qualification.
