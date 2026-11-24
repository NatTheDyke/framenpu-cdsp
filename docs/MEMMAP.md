# CDSP shared-buffer and VTCM map

This documents the verified address map of the 32 MiB CDSP DMA buffer and the
8 MiB VTCM on the one supported firmware (SteamOS 0.5.8 / CDSP ABI 9 /
Hexagon v75). It exists to locate a **persistent arena** for resident weights,
activations and kernel code that does not collide with Valve's gaze and
FeatureNN regions. All offsets are relative to the shared buffer base; the DSP
sees the buffer at cached VA `0x87000000` and uncached VA `0xf0000000`.

The map is static-analysis plus on-device evidence, not a Qualcomm memory
specification. It applies only to the firmware hash pinned in
[profile.py](../src/framenpu_cdsp/profile.py).

## Address translation (TLB)

A read-only worker-0 probe ([tlb_probe.c](../tests/tlb_probe.c), driven by
[hardware_memmap.py](../tests/hardware_memmap.py)) dumped the 12 live global TLB
entries. Every entry is read/write/execute; there is **no page-level protection**
separating clients, so region ownership is by firmware convention only.

| DSP virtual range | Backs | Attr | Notes |
|---|---|---|---|
| `0x87000000`–`0x88ffffff` | whole 32 MiB buffer | cached | firmware's normal view |
| `0xf0000000`–`0xf1ffffff` | whole 32 MiB buffer | uncached | the alias the host and jobs use today |
| `0x31000000`–`0x317fffff` | VTCM, 8 MiB | cached | all of VTCM is mapped |
| `0x30000000` (1 MiB), `0x32000000` (16 MiB) + others | device | device | `0x32000000` includes HMX clock/power |

Consequences: kernels can use the **cached** alias for operands (with explicit
cache maintenance at the host boundary) instead of the slow uncached alias, and
the 4 KiB code reservation is a library convention, not a hardware limit.

## Buffer usage

Derived from Ghidra value-flow of the worker dispatch (`FUN_87000100`) and the
per-worker region setup (`FUN_87005330`), corroborated by host snapshots. The
FeatureNN regions are **computed** as `base + HTID*stride`, so a scan for
absolute address constants cannot see them — an earlier constant-only scan
wrongly reported the top 9.7 MiB as free.

| Offset range | Size | Contents |
|---|---|---|
| `0x000000`–`0x58e000` | 5.6 MiB | firmware image (code + data) |
| `0xa90000`–`0xb23000` | ~0.6 MiB | identity, control block, worker control |
| `0xb22000`–`0x1a22000` | 3 × 5 MiB | **per-worker FeatureNN slots**, `0x87b22000 + HTID*0x500000`, workers 0/1/2 |
| `0x1522000`–`~0x1543000` | — | gaze (HTID 3) control + IO; shares worker-2's slot base, time-multiplexed |
| `0x1647080`–`0x164709c` | — | live firmware control flags (worker-exit sentinel, gaze/exit) |
| **`0x1a22000`–`0x2000000`** | **5.87 MiB** | **unused — persistent arena region** |

Worker dispatch roles (from `FUN_87000100`): **HTID 4** is the master CPU-request
loop (handles operation selectors 2/4/8/0x20); **HTID 3** is gaze; other HTIDs are
worker threads that run `FUN_87005330` when woken. The FeatureNN worker count is a
runtime field in the request (`control+0x88` low 7 bits → resume mask), with 3
(workers 0/1/2) the observed maximum. A custom protocol should bound this itself
rather than trust the field.

VTCM is tight and **not** a free-arena source: per-worker slots are
`0x31000000 + HTID*0xf0000` (960 KiB) and `0x30000000 + HTID*0x80000` (512 KiB);
FeatureNN also uses `0x311e0000`/`0x31492080`; gaze uses `0x31700000`–`0x31776600`.
The first 8 KiB of VTCM that HMX jobs borrow today is not proven free and is only
safe while the stock operation gate is held.

Incidental corroboration: FeatureNN normalizes its input as `byte * 1/255`
(`0.003921569`), matching FrameTop's grayscale-to-[0,1] step in `nets.cpp`.

## Persistent arena: `0x1a22000`–`0x2000000` (5.87 MiB)

This span sits above every reserved per-worker slot. Two independent lines of
evidence support treating it as owned model storage:

1. **Static:** no firmware code references any address at or above `0x1a22000`
   (constants or computed `base + HTID*stride`).
2. **Empirical soak:** [host_canary.py](../tests/host_canary.py) filled the whole
   span with a deterministic offset-keyed pattern; the headset was then worn and
   actively used for several minutes (gaze, hand tracking, controllers); readback
   showed **0 disturbed bytes**, with the DSP service keeping its PID and zero
   restarts. On-device, the DSP also has verified TLB read/write across the span
   (a cached-alias fill reached DDR as the host observed it).

Residual caveat: the soak observes DDR, so a region used purely as cache-resident
DSP scratch that never flushes would not be detected. Nothing is addressed above
`0x1a22000` statically, so this is corroboration rather than the primary evidence.

Do **not** use `0x1648000`–`0x1a22000`: that is worker-2's FeatureNN slot and the
gaze overflow, and it holds the firmware's live worker-exit flags at `0x1647080+`.

## Coexistence note

With the stock tracking stack live, benchmark-backend submissions over the full
span repeatedly brushed the 40 ms host completion window, because they contend for
the single operation gate and workers 0/1/2 with real gaze/FeatureNN work (the DSP
finished correctly each time; only the host timed out). Sustained custom inference
alongside tracking will require the worker-0-only [direct dispatcher](DISPATCH.md)
or a cooperative scheduling model, not the stock benchmark path. The soak fill
itself used a **gate-free host write** so it never perturbed live tracking.

## Reproduction

~~~sh
# Build the read-only probes on a workstation
framenpu-cdsp build-job tests/tlb_probe.c    --output build/tlb_probe.fjob
framenpu-cdsp build-job tests/canary_probe.c --output build/canary_probe.fjob

# On the Frame (as steamos)
python tests/hardware_memmap.py --run-on-device --probe tlb_probe.fjob --count 64

python tests/host_canary.py --phase fill   --seed 0x6af91db2   # then wear + use the headset
python tests/host_canary.py --phase verify --seed 0x6af91db2
~~~

The canary writes only at or above the verified `0x1a22000` floor and never
touches firmware memory. The host-only path does not claim the operation gate.
