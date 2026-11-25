"""The Mercury image on the CDSP: image validation, arena ownership, staging and submission.

This is the device half of a Mercury session (framenpu.mercury_runtime is the model half). A
MercuryLease owns the shared window, the image reservation and its seal from open() to close();
it stages prepared requests through the session's hooks, submits one DSP job per call (one
request, a --multi frame, or a --stream frame) and checks every completion before the outputs
are read. Any failed check raises RecoveryRequired and leaves the device bytes in place.

Requests are duck-typed: ``count``, ``sl`` (the model: ``plan``, ``slot``, ``tag``, ``crop_ms``),
``timing`` (set here), and ``restart()`` / ``mark(name)`` / ``marks`` for profiling.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import secrets
import struct
import threading
import time

from . import _mercury_common as C
from .device import Device
from .errors import InvalidModel, RecoveryRequired
from .job import Job
from .profile import PROFILE
from .resident import CONTROL, RECORD, SSR, arena_lease

# the qualified Mercury image shipped with this package (link.json, image, loader): 1bcbbdd8, the
# 4-crop streamed build (tools/build_teacher_link.py ... --max-crops 4 rebuilds it byte for byte)
BUNDLED_JOBS = Path(__file__).with_name("data") / "mercury"

IMAGE_SHA256 = "d7a78a81469c10acaf193b0d99cc0415f92d1de845a7a9aa292a7a1ebe691819"
# The same build with --mheads: the heatmap head (step 2c) runs on the DSP too.
HEADS_IMAGE_SHA256 = "582d79ce821304ab90b48d18c5e9d7c5e4396614b9c6eb70bf2e7b6febf114d3"
# The heads build with --resident: parameter images stay in the FeatureNN blob region.
# Built with --omit-fp --pw-pipe: no leaf frames on the DDR stack for the group barriers
# to wait on, overlapped 1x1 loops (same outputs bit for bit; 3.35 -> 2.96 ms DSP per crop).
# 9160e4c7 (16 KiB) was the --omit-fp --pw-pipe build. This one adds --hmx-inline --pool
# --tiles --post-batch --pw-deep --pw-lg 3 --code-kib 20: three workers, tile-layout 1x1
# inputs, 8-block 1x1 groups; same outputs bit for bit, 2.96 -> 1.63 ms DSP per crop.
RESIDENT_IMAGE_SHA256 = "5d15e715948564ee0546cfb05d5cabb5c767f6f302936dad99716c0b15637ee4"
# The same build with --res-slots: four residency tag slots, so several models (ft7 and the
# stereo 3D student) keep their parameter images resident side by side. Without the option the
# sources still build RESIDENT_IMAGE_SHA256 byte for byte.
SLOTS_IMAGE_SHA256 = "320ae5c3f856b91ea9cb8b99e0fe04159309624eba13a8fedd8d6d2c8cb1ae6c"
RES_SLOTS = 4
# The slots build with --cbias --win-wide --s2-rev: biases packed one word per channel pair
# (expanded per op at the top of the blob region), the window up to the image reservation
# (0x1ff0000), so the hand detector fits resident beside ft7, and the stride-2 column blocks
# stored last to first (the detector's 80 -> 40 layer is wider than one block; the post-batch
# order clobbered the next row's first columns, 382fe1e6 on device). Same loader.
DET_IMAGE_SHA256 = "bebc1ea8d758b3b77e46a820344983c8981d4ba288426b527a2b39bba686a705"
# The same build with --multi: a request may carry two segments (two models' tables, crop counts
# and residency slots, e.g. the detector and ft7 of one frame), run back to back in one DSP job
# under one gate hold and one 20 ms budget (infer_frame). One-segment requests run as before.
MULTI_IMAGE_SHA256 = "ffe783131158f302d28c68812c9d5c4855892dee01e3e4abb0ef18e5c145ec94"
MULTI_SEGMENTS = 2
# The --multi build with --stream: in a streamed request segment 1 starts once the host has
# staged it (request word 12 = the nonce in word 11), and each segment publishes meta word 13 =
# nonce after its outputs. infer_frame then prepares and stages segment 1 and reads and finishes
# segment 0 while the DSP runs; only segment 0's prepare and segment 1's finish stay serial.
STREAM_IMAGE_SHA256 = "cd656b5d4641ac37b1820549dacef93859412f42fb60fe093641907f4630e5de"
# The --stream build with --max-crops 4: a keypoint request may carry four crops (both hands in
# both views of one capture); everything else as STREAM_IMAGE_SHA256.
CROPS4_IMAGE_SHA256 = "1bcbbdd8c13540e5ff11d25355c7aac9f800008020e48a0ab19075d70a58c162"
STREAM_LATE = 0x7D  # the kernel's error when segment 1 was not staged within 5 ms
# Straight-line VTCM backup/restore (the uncached slot refetched each loop packet: 385 -> 42 us).
# The pool loader (workers 1 and 2 enter the image too) for the 20 KiB reservation.
LOADER_SHA256 = "56cbe6836d17bcad31948380a453e6408cf5660b9425aaa93aae9d0227743fd1"
CODE, CODE_BYTES, SEAL_AT, SEAL = 0x1FF0000, 0x4000, 0x1FF3F80, 0x544C4E4B
IMAGE_CACHED, SLOT_UNCACHED = 0x88FF0000, 0xF0A92000
ADMIN_MAGIC, ADMIN_OPCODE = 0x544C4144, 112


def image_kind(record):
    """(heads, resident) of a link record."""
    return bool(record.get("mheads")), bool(record.get("resident"))


def reservation(record):
    """(bytes, seal vector) of a link record's image reservation at CODE: the
    16 KiB default or the 20 KiB --code-kib 20 build. The loader's admin command
    rejects any other size, so a misstated record fails before sealing."""
    r = record.get("reservation", {})
    size = r.get("bytes")
    if (
        r.get("code") != f"{CODE:#x}"
        or size not in (0x4000, 0x5000)
        or (r.get("seal_vector") != f"{CODE + size - 128:#x}")
    ):
        raise InvalidModel("Mercury image reservation differs")
    return size, CODE + size - 128


def load_jobs(folder):
    """Accept the qualified replay build only (with or without the DSP heatmap
    head, and the heads build with resident parameters, with or without
    residency slots), including its existing
    loader. Returns (loader, image, heads); resident_image(folder) tells the rest."""
    folder = Path(folder)
    record = json.loads((folder / "link.json").read_text())
    loader = Job.load(folder / "teacher_link_loader.fjob")
    image = (folder / "teacher_link.bin").read_bytes()
    heads, resident = image_kind(record)
    if resident and not heads:
        raise InvalidModel("Mercury resident parameters need the heads image")
    slots = record.get("res_slots") == RES_SLOTS
    if record.get("res_slots") not in (None, RES_SLOTS) or (slots and not resident):
        raise InvalidModel("Mercury residency slots differ")
    wide = (
        bool(record.get("cbias")) or record.get("win_hi") is not None or bool(record.get("s2_rev"))
    )
    if wide and (
        not slots
        or not record.get("cbias")
        or record.get("win_hi") != "0x1ff0000"
        or not record.get("s2_rev")
    ):
        raise InvalidModel("Mercury packed-bias / wide-window build differs")
    multi, stream = record.get("multi"), record.get("stream")
    if (
        multi not in (None, MULTI_SEGMENTS)
        or (multi and not wide)
        or stream not in (None, True)
        or (stream and not multi)
    ):
        raise InvalidModel("Mercury multi-segment build differs")
    crops4 = record.get("max_crops")
    if crops4 not in (None, C.MAX_CROPS_WIDE) or (crops4 and not stream):
        raise InvalidModel("Mercury crop-count build differs")
    want = (
        (
            CROPS4_IMAGE_SHA256
            if crops4
            else STREAM_IMAGE_SHA256
            if stream
            else MULTI_IMAGE_SHA256
            if multi
            else DET_IMAGE_SHA256
            if wide
            else SLOTS_IMAGE_SHA256
            if slots
            else RESIDENT_IMAGE_SHA256
        )
        if resident
        else HEADS_IMAGE_SHA256
        if heads
        else IMAGE_SHA256
    )
    if (
        hashlib.sha256(image).hexdigest() != want
        or hashlib.sha256(loader.code).hexdigest() != LOADER_SHA256
        or record["linked"]["sha256"] != want
        or record["loader"]["sha256"] != LOADER_SHA256
        or loader.name != "teacher_link_loader_v1"
        or not loader.requires_hmx
        or not loader.supports_direct
        or record.get("pw_group") != 8
        or any(
            not record.get(k)
            for k in ("stride2", "s2_even", "s2_direct", "pw_fast", "dw_fast", "replay")
        )
        or any(record.get(k) for k in ("heads", "stem", "pw_profile"))
    ):
        raise InvalidModel("Mercury requires the qualified replay image and loader")
    return loader, image, heads


def _completion(result, output_bytes):
    """Check worker identity, code/EVB, status and fault fields under the gate."""
    if len(result.workers) != 3 or len(result.output) != output_bytes:
        raise RecoveryRequired("Mercury completion is incomplete")
    for i, row in enumerate(result.workers):
        if (
            len(row) != 16
            or not row[0]
            or row[0] != result.workers[0][0]
            or row[1] != i
            or row[7] != SSR[i]
            or any(row[2:6])
            or any(row[14:16])
        ):
            raise RecoveryRequired("Mercury worker completion differs")
    if (
        result.workers[0][8] != 0x87000000
        or result.kernel_reference_ticks is None
        or not 0 < result.kernel_reference_ticks < 768000
    ):
        raise RecoveryRequired("Mercury EVB or wrapper time differs")


def _completion_guard(check):
    """Any interrupted or failed integrity check retains the DSP resources."""

    def validate(result):
        try:
            return check(result)
        except RecoveryRequired:
            raise
        except BaseException as exc:
            raise RecoveryRequired("Mercury completion validation failed") from exc

    return validate


def replay_metadata(result, count, descriptors, image_bytes, resident=None):
    """resident: None for a non-resident image; else this request's mode (bool).
    A resident image reports meta[12] = mode | copied << 1 | tagged << 2, and a
    complete request tags exactly when it copied."""
    _completion(result, C.META_BYTES)
    return _segment_metadata(
        result.output, count, descriptors, image_bytes, resident, result.kernel_reference_ticks
    )


def frame_metadata(result, segments, image_bytes, resident, done=None):
    """A --multi request's metadata: segments = [(crops, descriptors), ...], one
    META_BYTES block each (segment k's in block k), the segments' bodies together
    within the request's one 20 ms budget and the wrapper's time."""
    _completion(result, C.META_BYTES * len(segments))
    out = [
        _segment_metadata(
            result.output[k * C.META_BYTES : (k + 1) * C.META_BYTES],
            count,
            n,
            image_bytes,
            resident,
            result.kernel_reference_ticks,
            done,
        )
        for k, (count, n) in enumerate(segments)
    ]
    if sum(t["crop_end_ticks"][-1] for t in out) > min(384000, result.kernel_reference_ticks):
        raise RecoveryRequired("Mercury segment bodies exceed the request budget")
    return out


def _segment_metadata(output, count, descriptors, image_bytes, resident, kernel_ticks, done=None):
    """done: a streamed request's nonce, which the segment publishes in word 13."""
    m = list(struct.unpack("<256I", output))
    if done is not None:
        if m[13] != done:
            raise RecoveryRequired("Mercury segment completion word differs")
        m[13] = 0
    status = m[12]
    if resident is not None:
        if status not in ((3 | 4, 1) if resident else (0,)):
            raise RecoveryRequired("Mercury residency status differs")
        m[12] = 0
    ticks = list(m[96 : 96 + descriptors])
    ends = list(m[48 : 48 + count]) if count > 1 else [ticks[-1]]
    base = ends[-2] if count > 1 else 0
    if (
        m[0] != C.RS_MAGIC
        or m[2] != descriptors * count
        or not IMAGE_CACHED <= m[1] < IMAGE_CACHED + image_bytes
        or any(m[3:48])
        or any(m[48 + count : 96] if count > 1 else m[48:96])
        or any(m[96 + descriptors :])
        or any(b <= a for a, b in zip([0] + ends, ends))
        or any(b < a for a, b in zip(ticks, ticks[1:]))
        or not base < ticks[0] <= ticks[-1] == ends[-1] <= min(384000, kernel_ticks)
    ):
        raise RecoveryRequired("Mercury replay metadata or budget differs")
    return {
        "crops": count,
        "params_copied": bool(status & 2) if resident else None,
        "crop_end_ticks": ends,
        "crop_body_ms": [(b - a) / 19200 for a, b in zip([0] + ends, ends)],
        "body_ms": ends[-1] / 19200,
    }


@dataclass(frozen=True)
class ImageFeatures:
    """What a validated Mercury image supports, from its link record (load_jobs has already
    checked the record against the pinned hash, so the two cannot disagree)."""

    sha256: str
    heads: bool
    resident: bool  # parameter images may stay in the FeatureNN blob region
    res_slots: bool  # RES_SLOTS residency tag slots: several models resident together
    cbias: bool  # packed biases and the wide window (RS_CBIAS + RS_WIN_WIDE)
    multi: bool  # MULTI_SEGMENTS requests per job (infer_frame)
    stream: bool  # segment 1 staged while segment 0 runs
    max_crops: int
    tiles: bool  # the record says RS_TILES
    deep8: bool  # the record says 8-block 1x1 groups (--pw-deep --pw-lg 3)
    code_bytes: int
    seal_at: int

    @property
    def win_hi(self):
        return C.WIN_HI_WIDE if self.cbias else C.WIN_HI


def image_features(image, record):
    heads, resident = image_kind(record)
    code_bytes, seal_at = reservation(record)
    if len(image) > seal_at - CODE:
        raise InvalidModel("Mercury image overlaps its seal vector")
    stream = bool(record.get("stream"))
    return ImageFeatures(
        sha256=hashlib.sha256(image).hexdigest(),
        heads=heads,
        resident=resident,
        res_slots=record.get("res_slots") == RES_SLOTS,
        cbias=bool(record.get("cbias")),
        multi=bool(record.get("multi")),
        stream=stream,
        max_crops=C.MAX_CROPS_WIDE if record.get("max_crops") else C.MAX_CROPS,
        tiles=bool(record.get("tiles")),
        deep8=bool(record.get("pw_deep")) and record.get("pw_lg") == 3,
        code_bytes=code_bytes,
        seal_at=seal_at,
    )


def open_image(folder):
    """(loader, image, ImageFeatures) of a qualified Mercury build folder."""
    loader, image, _ = load_jobs(folder)
    record = json.loads((Path(folder) / "link.json").read_text())
    return loader, image, image_features(image, record)


class MercuryLease:
    """Exclusive use of the shared window and image reservation for a Mercury session.

    models: the compiled plans (plugin.CDSPPlan: plan, slot, tag, crop_ms). Each request q
    carries sl (its model: plan, slot, tag, crop_ms), count and staged, the plan's staging for
    it: staged.stage(mem) writes its inputs, staged.read(get, k) reads and checks crop k's
    outputs (get(at, n) -> bytes). resident: the request mode on a resident image.
    """

    def __init__(
        self,
        loader,
        image,
        features,
        models,
        resident,
        device_factory=Device,
        device_options=dict,
        lease_factory=arena_lease,
    ):
        self.loader, self.image, self.features = loader, image, features
        self.image_sha256 = features.sha256
        self.code_bytes, self.seal_at, self.win_hi = (
            features.code_bytes,
            features.seal_at,
            features.win_hi,
        )
        self.multi, self.stream = features.multi, features.stream
        self.resident_image = features.resident
        self.resident = bool(resident) if features.resident else False
        self.models = models
        self._device_factory, self._lease_factory = device_factory, lease_factory
        self._device_options = device_options  # per device session: extra Device keywords
        self.device = self.arena = self.power = None
        self.closed = self.owned = self.sealed = False
        self.marker = RECORD + secrets.token_bytes(16) + bytes.fromhex(self.image_sha256)
        self.saved = []
        self.lock = threading.RLock()

    def _read_outputs(self, q):
        mem = self.device._transport.memory
        get = lambda at, n: bytes(mem[at : at + n])  # noqa: E731
        for k in range(q.count):
            if not q.staged.read(get, k):
                raise RecoveryRequired("Mercury output has invalid padding or values")

    @staticmethod
    def _stage(mem, q):
        q.staged.stage(mem)

    def _owner(self):
        if (
            bytes(self.device._transport.memory[CONTROL : CONTROL + len(self.marker)])
            != self.marker
        ):
            raise RecoveryRequired("Mercury arena ownership changed")

    def _reservation(self, sealed):
        mem = self.device._transport.memory
        expect = bytearray(self.image.ljust(self.code_bytes, b"\0"))
        struct.pack_into("<I", expect, self.seal_at - CODE, SEAL if sealed else 0)
        if bytes(mem[CODE : CODE + self.code_bytes]) != expect:
            raise RecoveryRequired("Mercury image or seal changed")
        self._owner()

    def _gate(self, action):
        t = self.device._transport
        t.claim(self.device.claim_timeout)
        try:
            action(t.memory)
        except RecoveryRequired:
            t.poisoned = True
            raise
        finally:
            if not t.poisoned:
                t.release_unsubmitted()

    def _admin(self, command):
        before, after = self.sealed, command == 7

        def validate(result):
            _completion(result, 256)
            m = struct.unpack("<64I", result.output)
            if (
                m[0] != ADMIN_MAGIC
                or not SLOT_UNCACHED <= m[1] < SLOT_UNCACHED + len(self.loader.code)
                or m[2] != (SEAL if after else 0)
                or any(m[3:])
            ):
                raise RecoveryRequired("Mercury seal/retire completion differs")
            self._reservation(after)
            self.sealed = after

        self.device._submit(
            self.loader,
            struct.pack("<4I", ADMIN_MAGIC, command, CODE, self.code_bytes).ljust(128, b"\0"),
            opcode=ADMIN_OPCODE,
            params=(0,),
            output_size=256,
            use_direct=True,
            _prepare=lambda t: self._reservation(before),
            _completion_validator=_completion_guard(validate),
            _held_power=self.power,
        )

    def open(self):
        if self.closed:
            raise RuntimeError("Mercury session is closed")
        if self.device is not None:
            return
        self.arena = self._lease_factory()
        try:
            self.device = self._device_factory(dispatch="direct", **self._device_options())
            self.device.__enter__()

            def install(mem):
                if bytes(mem[CONTROL : CONTROL + len(RECORD)]) == RECORD:
                    raise RecoveryRequired("resident refusal record present")
                self.saved = [
                    (at, bytes(mem[at : at + size]))
                    for at, size in (
                        (C.WIN_LO, self.win_hi - C.WIN_LO),
                        (CODE, self.code_bytes),
                        (CONTROL, 128),
                    )
                ]
                self.owned = True
                mem[CONTROL : CONTROL + len(self.marker)] = self.marker
                image = self.image.ljust(self.code_bytes, b"\0")
                mem[CODE : CODE + self.code_bytes] = image
                for sl in self.models:
                    for off, data in C.setup_writes(sl.plan.plan).items():
                        mem[off : off + len(data)] = data
                        if bytes(mem[off : off + len(data)]) != data:
                            raise RecoveryRequired("Mercury parameter upload differs")
                self._reservation(False)

            self._gate(install)
            self._admin(7)
        except BaseException:
            self.close()
            raise

    def run_frame(self, ps, during=None):
        """Stage, run and read back prepared requests as one DSP job: one request, or on a
        --multi image up to MULTI_SEGMENTS of distinct models (segment k = ps[k])."""
        if len(ps) > 1 and not (self.multi and len(ps) <= MULTI_SEGMENTS):
            raise ValueError("this image runs one request per job")
        p = ps[0]
        p.restart()
        self.open()
        if self.power is None:
            self.power = self.device._hold_power()
        p.mark("open_power_ms")

        def stage(mem):
            self._reservation(True)
            for q in ps:
                self._stage(mem, q)

        self._gate(stage)
        p.mark("gate_stage_ms")
        jobs = [q.sl.plan.plan.jobs[0] for q in ps]
        if len(ps) > 1 and any(
            j.first or j.end != len(q.sl.plan.plan.table) // 128 for j, q in zip(jobs, ps)
        ):
            raise ValueError("a segment runs its whole table")  # the kernel runs segment 1 whole
        job = jobs[0]
        phases = {} if p.marks is not None else None
        resident = self.resident if self.resident_image else None

        def validate(result):
            self._reservation(True)
            if len(ps) == 1:
                p.timing = replay_metadata(
                    result, p.count, job.end - job.first, len(self.image), resident=resident
                )
            else:
                for q, t in zip(
                    ps,
                    frame_metadata(
                        result,
                        [(q.count, j.end - j.first) for q, j in zip(ps, jobs)],
                        len(self.image),
                        resident,
                    ),
                ):
                    q.timing = t
            for q in ps:
                self._read_outputs(q)

        self.device._submit(
            self.loader,
            b"".join(self._payload(q.count, q.sl) for q in ps),
            opcode=C.RS_OPCODE,
            params=(1, job.first, job.end, 0, 0, 0),
            output_size=C.META_BYTES * len(ps),
            use_direct=True,
            _prepare=lambda t: self._reservation(True),
            _completion_validator=_completion_guard(validate),
            _held_power=self.power,
            _expect_s=0.8e-3 * sum(q.sl.crop_ms * q.count for q in ps),
            _phase_times=phases,
            _while_waiting=during,
        )
        p.mark("submit_ms")
        if phases is not None:
            p.marks["submit"] = phases
        for q in ps:
            q.sl.crop_ms = q.timing["body_ms"] / q.count

    def run_streamed(self, p0, sl1, feeds1, prepare, finish):
        """A --stream frame of two segments: p0 (prepared) is staged before the job; while the DSP
        runs it, the waiting thread prepares and stages segment 1 (sl1, feeds1), releases it (word
        12 = nonce, a seq-cst store after the staging), then waits for segment 0's completion word
        and reads and finishes its outputs. Returns (segment 1's pending request, segment 0's
        outputs or None when segment 0 had not published in time: then they are read after the
        job like any other). A failure of the early read or finish is left to the normal path."""
        if sl1.slot == p0.sl.slot:
            raise ValueError("a frame's requests are of distinct models")
        t = self.device._transport
        nonce = 0
        while not nonce:
            nonce = struct.unpack("<I", secrets.token_bytes(4))[0]
        ready_at = PROFILE.arena + 128 + 48  # segment 1's request word 12
        done_at = PROFILE.arena + 256 + 13 * 4  # segment 0's meta word 13 (output at 256)
        box = {"p1": None, "early": None, "read0": False, "error": None}
        p0.restart()
        self.open()
        if self.power is None:
            self.power = self.device._hold_power()
        p0.mark("open_power_ms")

        def stage0(mem):
            self._reservation(True)
            self._stage(mem, p0)

        self._gate(stage0)
        p0.mark("gate_stage_ms")
        job0, job1 = p0.sl.plan.plan.jobs[0], sl1.plan.plan.jobs[0]
        for j, sl in ((job0, p0.sl), (job1, sl1)):
            if j.first or j.end != len(sl.plan.plan.table) // 128:
                raise ValueError("a segment runs its whole table")
        phases = {} if p0.marks is not None else None
        resident = self.resident if self.resident_image else None

        def during():
            try:
                p1 = prepare(sl1, feeds1)
                box["p1"] = p1
                self._reservation(True)
                self._stage(t.memory, p1)
            except BaseException as exc:  # raised after the job, which still completes cleanly
                box["error"] = exc
            # after the staging (seq-cst); on a failure too, so segment 1 runs (on whatever the
            # window holds: its outputs are discarded) instead of stalling the gate for 5 ms
            t.put(ready_at, nonce)
            if box["error"] is not None:
                return
            # segment 0 publishes after its outputs; bounded: else read after the job
            end = time.monotonic() + 2e-3 + 3e-3 * p0.sl.crop_ms * p0.count
            while t.get(done_at) != nonce:
                if time.monotonic() > end:
                    return
                time.sleep(5e-5)
            try:
                self._read_outputs(p0)
                box["read0"] = True
                box["early"] = finish(p0)
            except RecoveryRequired:
                box["read0"] = False  # validate reads it again and decides
                box["early"] = None

        def validate(result):
            self._reservation(True)
            if box["error"] is not None:
                _completion(result, 2 * C.META_BYTES)  # the job itself completed cleanly
                if isinstance(box["error"], RecoveryRequired):
                    raise box["error"]
                return
            p1 = box["p1"]
            t0, t1 = frame_metadata(
                result,
                [(p0.count, job0.end), (p1.count, job1.end)],
                len(self.image),
                resident,
                done=nonce,
            )
            p0.timing, p1.timing = t0, t1
            if not box["read0"]:
                box["early"] = None
                self._read_outputs(p0)
            self._read_outputs(p1)

        def payload(q_count, sl):
            w = bytearray(self._payload(q_count, sl))
            struct.pack_into("<3I", w, 40, 1, nonce, 0)
            return bytes(w)

        count1 = len(feeds1)
        self.device._submit(
            self.loader,
            payload(p0.count, p0.sl) + payload(count1, sl1),
            opcode=C.RS_OPCODE,
            params=(1, job0.first, job0.end, 0, 0, 0),
            output_size=2 * C.META_BYTES,
            use_direct=True,
            _prepare=lambda t: self._reservation(True),
            _completion_validator=_completion_guard(validate),
            _held_power=self.power,
            _expect_s=0.8e-3 * (p0.sl.crop_ms * p0.count + sl1.crop_ms * count1),
            _phase_times=phases,
            _while_waiting=during,
        )
        if box["error"] is not None:
            raise box["error"]
        p0.mark("submit_ms")
        if phases is not None:
            p0.marks["submit"] = phases
        p1 = box["p1"]
        for q in (p0, p1):
            q.sl.crop_ms = q.timing["body_ms"] / q.count
        if box["early"] is not None and phases is not None:
            p0.timing["phases_ms"] = p0.marks
        return p1, box["early"]

    def _payload(self, count, sl=None):
        """Request words; a resident image adds the tag (words 5..8) and mode (word 9,
        1 | slot << 1 on a --res-slots image)."""
        sl = self.models[0] if sl is None else sl
        words = bytearray(C.payload(sl.plan.plan, count))
        if self.resident_image:
            mode = int(self.resident) | (sl.slot << 1 if self.resident else 0)
            struct.pack_into("<5I", words, 20, *sl.tag, mode)
        return bytes(words)

    def idle(self):
        """Release held clocks when tracking has no observations; keep the plan."""
        with self.lock:
            if self.power is not None:
                power, self.power = self.power, None
                if self.device._transport.poisoned:
                    power.abandon()
                else:
                    self.device._release_power(power)

    def close(self):
        with self.lock:
            if self.closed:
                return
            try:
                self.idle()
                if self.device is not None and self.owned and not self.device._transport.poisoned:
                    if self.sealed:
                        self._admin(8)

                    def restore(mem):
                        self._reservation(False)
                        for off, data in self.saved:
                            mem[off : off + len(data)] = data
                            if bytes(mem[off : off + len(data)]) != data:
                                raise RecoveryRequired("Mercury arena restoration differs")
                        self.owned = False

                    self._gate(restore)
            except BaseException:
                if self.device is not None and self.owned:
                    self.device._transport.poisoned = True
                raise
            finally:
                try:
                    if self.device is not None:
                        self.device.close()
                finally:
                    if self.arena is not None:
                        self.arena.close()
                    self.closed = True
