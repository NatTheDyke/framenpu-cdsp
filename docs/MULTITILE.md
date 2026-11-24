# Two depth tiles per job

This page records the initial scalar-packing two-tile implementation and its
timings. The subsequent [HVX packing change](HVX.md) retains this API and
execution bound, with a separate validation and timing comparison.

The opt-in `tiles_per_job=2` argument to `Device.matmul` combines up to two
depth-32 HMX products for one output tile. It requires direct dispatch:

~~~python
from framenpu_cdsp import Device

with Device(dispatch="direct") as npu:
    product = npu.matmul([[1.0] * 64], [[1.0] for _ in range(64)], tiles_per_job=2)
print(product)  # [[64.0]]
~~~

The default remains `tiles_per_job=1`. Model files and `Model.run` retain their
existing behavior; this first implementation is an explicit matrix API option.
Ordinary matrix dimensions remain bounded to 256. Output rows and columns are
still scheduled in blocks of 32, while the new path processes up to 64 reduction
features per job. Each such job uploads its data, runs its products under one
HMX power cycle, and restores its borrowed spans before acknowledgment.

## Accumulation and bounds

Each depth-32 product retains the existing FP16 rounding. The DSP adds those
products to an FP32 output in reduction order. For larger K, the next chunk
receives the previous FP32 output as its initial accumulator and continues
that same order. Bias and optional ReLU apply only in the final chunk.

This avoids reassociating partial sums and supports intermediate accumulators
above FP16 range. For example, an intermediate sum of 131,008 is carried as
FP32. The host still downloads/uploads that seed between chunks; weights and
intermediates do not acquire persistent memory ownership.

Only depth tiling changes. Copies, matrix packing, the 8 KiB VTCM reservation,
the 128 KiB job arena and the dispatch protocol remain as before. The context
now records direct dispatch in flag bit 1. This lets the kernel reject multi-tile
work on the benchmark backend, whose arithmetic would consume additional time.
The 40 ms publication-to-completion timeout remains in force.

## Built-in opcode 6

Parameters are `(M, K, N, relu, seeded)`; the remaining parameter word is unused.
The kernel requires HMX and direct flags, `1 <= M,N <= 32`, `1 <= K <= 64`,
and Boolean `relu`/`seeded`. Thus at most two HMX products execute in the job.
Invalid shapes, flags and byte lengths fail before kernel operand access.

Input contains row-major FP16 A and B, alignment padding, FP32 bias and an
optional row-major FP32 seed:

~~~text
bias_offset = align4(2 * (M*K + K*N))
input_bytes = bias_offset + 4*N + (seeded ? 4*M*N : 0)
output_bytes = 4*M*N
~~~

The largest chunk with a seed uses 16,512 combined input/output arena bytes.
Opcode 1 keeps its existing single-depth behavior. The host rejects a tile
budget other than 1 or 2, or two-tile mode with benchmark dispatch, before
connecting to hardware.

## Device validation — 2026-11-14

The 3,380-byte runtime passed the existing single-tile numerical suites on both
backends, including all five padded/tiled GEMMs and the saved XOR model. The
new path then passed six shapes, including `(32,128,32)`, `(33,65,35)` and
`(1,256,1)`, plus FP32 rounding/order, seed-range and final bias/ReLU checks.
Six malformed or unsupported raw requests were rejected and the session
remained reusable.

The multi-tile correctness run audited restoration for 34 submissions, including
30 HMX-marked jobs. All borrowed DDR spans, callback and HMX configuration
matched their gate-acquisition values before acknowledgment. Final workers had
baseline SSR values; the dispatcher and first 112 KiB of firmware code were
restored. The DSP service retained its PID with zero restarts.

Timing used the same 32x64x32 multiply, bias and ReLU on direct dispatch, with
100 warm samples per mode after one first call. Restoration audits were
disabled during timing:

| Measurement | One tile per job | Two tiles per job |
|---|---:|---:|
| Full matmul wall-time p50 | 60.886 ms | 38.288 ms |
| Full matmul wall-time p99 | 70.504 ms | 49.117 ms |
| Public submissions per matmul | 4 | 1 |
| HMX power cycles per matmul | 2 | 1 |
| Maximum measured job publication-to-completion | 15.489 ms | 26.013 ms |

The median improved about 37.1%. Full-call times include Python serialization,
uploads, power setup/teardown and cleanup; the 40 ms job timeout starts after
power setup and ends at observed completion. Neither these samples nor the
two-tile cap establish hard real-time scheduling. Sustained tracking, thermals
and eye-tracking coexistence remain unmeasured.

The sanitized [numerical summary](../evidence/multitile-validation.json) records
payload identity, cases, timing and final state. Reproduce on the supported
Frame with an explicit hardware opt-in:

~~~sh
sudo .venv/bin/python tests/hardware_multitile.py --run-on-device --correctness-only
sudo .venv/bin/python tests/hardware_multitile.py --run-on-device --iterations 100
~~~
