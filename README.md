# framenpu-cdsp

**The Steam Frame CDSP device backend for [framenpu](https://github.com/NatTheDyke/framenpu):
custom Hexagon jobs, HMX inference and the Mercury hand-tracking image.**

The package registers the `cdsp` plugin that `framenpu.open(..., "device")` and `framenpu serve`
load ([contracts](https://github.com/NatTheDyke/framenpu/blob/main/docs/CONTRACTS.md)). It also
keeps the early API it grew from: a Python host API for custom jobs, a freestanding Hexagon v75 C
SDK and a small data-only model format.

This is an alpha research library. It supports one exact firmware profile and does not isolate
native jobs. Read [compatibility](#compatibility-and-limits) before device use.

## Install from a checkout

With a framenpu checkout beside this one:

~~~sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ../framenpu -e ".[dev]"
~~~

Python 3.11+ is required; framenpu and NumPy are the only dependencies. LLVM is needed only to rebuild or
compile kernels. Installing, inspecting models, and running offline tests require no headset or
root access.

## Create and inspect a model without hardware

~~~sh
python examples/make_xor.py
framenpu-cdsp inspect-model examples/xor.fnpu --batch-size 4
framenpu-cdsp inspect-ncnn examples/dense.param
framenpu-cdsp plan-resident examples/resident_residual.json
~~~

The synthetic examples are included. Model inspection and ncnn graph inventory
do not execute DSP code. The ncnn command is an inventory tool, not an importer.
The resident planner checks a storage schedule, tensor lifetimes and DDR bounds
without device access; see the [resident design](docs/RESIDENT.md). Its output
does not execute a model.

Supported model layers are Dense, ReLU and Clip:

~~~python
from framenpu_cdsp import Dense, Model

Model(
    2,
    [
        Dense([[1, -1], [-1, 1]], activation="relu"),
        Dense([[1], [1]]),
    ],
    name="xor",
).save("xor.fnpu")
~~~

Weights are FP16 [input_features, output_features]; biases and outputs are FP32.
The loader checks supported operators, dimensions, byte lengths and SHA256.
FNPU files contain declarative data, not executable kernels or pickle.

## Run on a supported Frame

Install the package in a virtual environment on the headset. Use the
interpreter from that environment when requesting root for HMX setup:

~~~sh
.venv/bin/python -m framenpu_cdsp info
sudo .venv/bin/python examples/run_xor.py
.venv/bin/python examples/run_custom.py
~~~

The examples locate their bundled files relative to the scripts, not the
working directory. Override them with --model or --job if needed.

Plain custom DSP jobs use the existing steamos service access. HMX setup needs
root access to /dev/mem. Unknown firmware is rejected before job writes.

~~~python
from framenpu_cdsp import Device, Model

model = Model.load("xor.fnpu")
with Device() as npu:
    output = model.run(npu, [[0, 0], [0, 1], [1, 0], [1, 1]])
print(output)  # [[0.0], [1.0], [1.0], [0.0]]
~~~

## Matrix and custom-kernel APIs

~~~python
from framenpu_cdsp import Device

with Device() as npu:
    product = npu.matmul([[1, 2, 3]], [[1], [2], [3]])  # [[14.0]]
    clipped = npu.elementwise([[-2, 3, 9]], activation="clip", maximum=6)
    summed = npu.add([[1, 2], [3, 4]], [[10, 20]])  # row broadcast
~~~

Direct dispatch also supports `npu.matmul(a, b, tiles_per_job=2)`, combining
two depth tiles per submission and preserving FP32 accumulation between
chunks. In a 32x64x32 device test, median wall time fell from 60.89 ms to
38.29 ms, with four submissions reduced to one. This option keeps the 40 ms
execution timeout and is opt-in; see [MULTITILE.md](docs/MULTITILE.md).

Full aligned tiles now use HVX packing automatically. A same-request comparison
reduced two-depth-tile 32x64x32 job publication-to-completion p50 from 25.28 ms
to 15.79 ms. Padded edges retain scalar packing. See [HVX.md](docs/HVX.md) for
correctness checks, timing boundaries and reproduction.

HVX output unpacking/conversion now applies to every supported GEMM tile shape,
while retaining the existing scalar FP32 accumulation. A subsequent comparison
reduced the same two-depth-tile job interval from 15.82 ms to 6.99 ms; see
[READBACK.md](docs/READBACK.md).

Implement [the C job ABI](src/framenpu_cdsp/sdk/include/framenpu.h), as demonstrated
by [vector_add.c](examples/vector_add.c), and compile on your workstation:

~~~sh
framenpu-cdsp build-job examples/vector_add.c --output build/vector_add.fjob
~~~

Copy the FJOB to your Frame and submit it:

~~~python
import struct
from framenpu_cdsp import Device, Job

job = Job.load("vector_add.fjob")
with Device() as npu:
    result = npu.submit(
        job, struct.pack("<8f", 1, 2, 3, 4, 5, 6, 7, 8), output_size=16, params=(4,)
    )
print(struct.unpack("<4f", result.output))  # (6, 8, 10, 12)
~~~

The compiler needs Hexagon-capable clang, ld.lld and llvm-objcopy. LLVM 23.1.4
produced the included payloads. Select executable paths with CLANG, LD_LLD and
LLVM_OBJCOPY; a standard-library/compiler runtime is not linked into the job.
The wheel includes SDK sources, the header, entry wrapper and linker script.

Native jobs are trusted supervisor code, not sandboxed model data. See
[ABI.md](docs/ABI.md) and [SECURITY.md](SECURITY.md).
The SDK now restores both SP and FP after guarded nested-C-frame faults.
Rebuild older custom FJOBs to receive the fix; see
[fault-unwind qualification](docs/FAULT_UNWIND.md).

## Compatibility and limits

| Item | Current support |
|---|---|
| Hardware | SM8650 / Hexagon v75 Steam Frame |
| Tested firmware | SteamOS 0.5.8, CDSP ABI 9; exact hashes in [profile.py](src/framenpu_cdsp/profile.py) |
| Models | Sequential Dense, ReLU and Clip |
| Kernels | At most 4 KiB native code; 128 KiB job arena |
| HMX scratch | First 8 KiB of VTCM, saved/restored |
| Matrix dimensions | Up to 256, scheduled in bounded depth/output-32 tiles |
| Arithmetic | FP16 depth-32 products, FP32 DSP accumulation across depth tiles |

The default backend executes a stock FLOPS benchmark per submission and scopes
HMX power to that job. This is unsuitable for real-time hand tracking.

An opt-in `Device(dispatch="direct")` prototype bypasses the benchmark using a
reversible 96-byte runtime dispatch edit. It keeps HMX/HVX on worker 0 and joins
all workers before completion. CPU/DSP validation passed: median warm empty-job
wall time fell from 4.29 ms to 0.73 ms across 100 samples per backend.
Both dispatch paths also pass padded/tiled HMX GEMMs and the saved XOR model;
median 32x32 GEMM submission wall time was 29.34 ms versus 25.36 ms with direct
dispatch. **Sustained tracking coexistence remains unvalidated.**
See [DISPATCH.md](docs/DISPATCH.md) for results, lifecycle, compatibility and
the explicit hardware comparison commands.

The stock dispatcher releases its completion/ownership window after roughly
50 ms even without acknowledgment; the host timeout is capped at 40 ms.
OS scheduling and other clients are not isolated. On RecoveryRequired, stop
using the session: possibly active payload/power are retained and deliberate
DSP/power recovery is required. The library never automatically restarts
tracking services. See the [failure contract](docs/ABI.md).

No proprietary firmware or hand-model weights are bundled. Power sequencing
is recovered from the exact firmware already installed on the device.

## Tests and reproducible builds

~~~sh
python -m unittest discover -s tests -p 'test_*.py' -v
ruff check .
ruff format --check .
python tools/check_repository.py
python tools/build_kernels.py --check
python -m build
python -m twine check dist/*
python tools/check_distribution.py --dist dist --framenpu path/to/framenpu-*.whl
~~~

Only the kernel rebuild/check step needs LLVM. The distribution check rebuilds
from the source archive, installs the wheel in a fresh environment, and checks
the CLI, models, SDK resources and examples without a headset.

The separate hardware test is opt-in and is never run by ordinary CI:

~~~sh
sudo .venv/bin/python tests/hardware_smoke.py --run-on-device
~~~

Configured CI covers Python 3.11–3.14. [Validation notes](docs/VALIDATION.md)
record 177 passing offline tests and distinguish local results from the
configured CI matrix.

## Hand tracking and contributions

The pinned FrameTop v20a landmark graph now has experimental streamed and
[resident DSP executors](docs/RESIDENT_LANDMARK.md). General ncnn import remains
unsupported; inspect-ncnn inventories
your own param file without execution. Bounded two-depth-tile jobs also have
validated HVX input packing and output conversion.

A [5.87 MiB persistent DDR region](docs/MEMMAP.md) is mapped and canary-checked;
an [offline resident allocation planner](docs/RESIDENT.md) now checks tensor
lifetimes, aliases and storage bounds. A [bounded resident GEMM proof](docs/RESIDENT_GEMM.md)
now loads weights once and streams changing inputs for 32×K×32 products,
K=32/64/128. Three hundred measured jobs plus checked warmups/reuse match all
325,632 output values; fault, replacement and teardown checks pass 372 audits.
Three [real-weight convolution layers](docs/CONVOLUTION.md) now pass pinned ncnn
comparisons on recorded infrared crops: the stem, a pointwise layer and K=64.
All 2,207,744 values pass the frozen arithmetic and total-error limits, including
held-out crops, with 3,558 restoration audits. Input quantization dominates the
stem's error. The current one-tile-per-request prototype takes several seconds
per layer. A [pinned complete-graph prototype](docs/LANDMARK.md) now adds HVX
depthwise/pooling/residual kernels and tensor aliases through bounded streamed
submissions. All 77 layers, 90 blobs and four outputs pass on four recorded
crops under frozen provisional limits: 20,351,360 layer-output values and
25,455 restoration audits. The streamed version keeps intermediates on the host
and takes 71–79 seconds per crop with audits. The subsequent resident executor
keeps weights and FP32 intermediates in the verified arena, uploads weights once
and streams changing inputs. Its uncached depthwise/submission costs are higher;
useful tracking performance and broader accuracy qualification remain open.
[Depthwise gather optimization](docs/DEPTHWISE_GATHER.md) now reduces audited
resident frames from about 326 to 184–187 seconds, preserving every head bit.
Depthwise native time improves 2.38× and its request count falls 54.5%; the
complete-frame improvement is 1.76×.
The opt-in [resident cache entry](docs/RESIDENT_CACHE.md) now executes kernel
instructions through the cached alias while preserving uncached data and the
existing ownership, work caps and timeout limits. Four-crop audited frames now
take 153.9–154.6 seconds, a further 17.0% time reduction, with identical heads.
The default stays uncached.
[Linked resident operators](docs/RESIDENT_LINK.md) then place all compute
operators in one image in the owned arena and let a request cross layers up to
a tick budget. With operator state and staged rows read from saved VTCM, a
frame needs 52 requests instead of about 10,960 and takes 1.66–1.69 seconds
at a 20 ms budget, with identical heads. Opt-in session-held HMX power removes
the per-request power cycling and brings a frame to 1.36–1.39 seconds.
An opt-in [depthwise reuse build](docs/DEPTHWISE_REUSE.md) caches broadcast
coefficients and shares product masks. A sequential pilot reduces depthwise
time 7.9% and frame time 6.1%, with 50 requests and bit-identical heads.
On top of it, opt-in [row tiling](docs/DEPTHWISE_TILE.md) of the 7- and 14-wide
depthwise layers brings depthwise time from 421 to 291 ms and a frame to
1.08–1.19 seconds in 43 requests.
An opt-in [fused 1x1 convolution path](docs/CONV_FAST.md) then adds bias and
clips inside the HMX unit and brings a frame to 0.67–0.77 seconds in 27
requests, with bit-identical heads.
The later [HMX model feasibility study](docs/HMX_MODEL_FEASIBILITY.md) uses
public weights and distinct synthetic operands to evaluate a smaller fused
architecture. Packed streaming dominates the measured pipeline; a half-width
candidate forecasts 47 ms of convolution work before gathering and other
operators. Reuse, a complete new backend and training/accuracy validation
remain necessary for the 17–25 ms native target.
The [accumulator isolation tests](docs/HMX_ACCUMULATOR.md) show shared HMX state
across workers with the recovered primitives. Keep each complete clear/load/store
transaction under one HMX owner; use HVX workers for other work.

The [correct resident GEMM baseline](docs/HMX_BASELINE.md) matched every output
across 400 measured jobs and 16 warmups, with a 7.41–7.44 GMAC/s load slope.
That uncached loop was limited by instruction fetch. A subsequent
[cached-code comparison](docs/HMX_CACHE.md), at unchanged clock configuration,
passed 800 measured jobs and 32 warmups with exact outputs. For four deep
resident transactions, HMX p50 fell from 9.51 ms to 56.25 us; the fitted cached
load slope is 1,264–1,269 GMAC/s. A separate
[two-block-load diagnostic](docs/HMX_BATCHING.md) passed another 800 measured
jobs and 32 warmups, with about 2.52x improvement in its uncached loop.

These remain experimental fixtures; the general matrix API still uses its
existing uncached path. The experiments reuse blocks within a job and do not
establish full-model throughput. Cached publication/cleanup costs roughly
1.5 ms per experimental job, and fresh sessions in that experiment take about
33 ms. The resident proof qualifies a fixed ownership/cache contract; its
commands still take roughly 12.4–12.6 ms median with per-request power and
full-window cleanup. Real convolution is now numerically qualified in the bounded
proof. Complete graph correctness now also passes in the streamed executor.
Next are bounded submission overhead and uncached data-access reductions while
preserving the frozen rules. See
[HAND_TRACKING.md](docs/HAND_TRACKING.md) for the roadmap and
[HMX_THROUGHPUT.md](docs/HMX_THROUGHPUT.md) for research context.

The optional [banded teacher path](docs/TEACHER_STEM.md) now runs from normalized
RGB through all four heads in two DSP requests, returning 512 bytes. Four-crop
qualification passes the existing accuracy ceilings. Median DSP time is
32.44 ms total; host wall through the audited Python harness is 56.75 ms.

[CONTRIBUTING.md](CONTRIBUTING.md) explains development and kernel changes.
The project is [MIT-licensed](LICENSE); see
[third-party provenance](THIRD_PARTY_NOTICES.md). This independent research
project is not endorsed by Valve or Qualcomm.
