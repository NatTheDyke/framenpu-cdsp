"""The CDSP device plugin: what framenpu sees of this backend (framenpu docs/CONTRACTS.md, plugin ABI 1).

Registered as ``cdsp`` under the ``framenpu.devices`` entry-point group; framenpu loads it with
``framenpu.devices.load("cdsp", jobs=...)`` and never imports this package directly.

A plugin is one opened Mercury image: construction loads and validates the image folder (the
bundled one when jobs is None) against the pinned hashes, and capabilities() reports what its link
record says it supports. Code branches on those fields, never on which pinned hash matched.

Clock settings are the plugin's, so a service turns them on in its unit's environment without
any framenpu option, and each device session the lease opens gets them from here:
- axi_mhz: the job-scoped AXI-master raise (axi.py), off unless set; default FNPU_AXI_MHZ.
  capabilities() then lists "axi".
- hmx_mhz: the HMX clock while HMX is held (power.py HMX_RATES); default FNPU_HMX_MHZ, else
  the 192 MHz bring-up rate. A faster rate is listed as "hmx_fast".
"""

from __future__ import annotations

import logging
import secrets
import struct
from dataclasses import dataclass, field
from typing import Any

from framenpu.contracts import Capabilities, Refusal

from . import _layout_native
from . import _mercury_common as C
from . import _mercury_layout as L
from . import staging as S
from .axi import AXI_RATES, AxiClock, rate_from_env
from .power import DEFAULT_HMX_MHZ, HmxPower
from .power import _rate as hmx_rate
from .device import Device
from .errors import InvalidModel
from .mercury_device import BUNDLED_JOBS, MULTI_SEGMENTS, RES_SLOTS, MercuryLease, open_image
from .resident import arena_lease

# ImageFeatures flags reported in Capabilities.features
FEATURE_FLAGS = ("heads", "resident", "res_slots", "cbias", "multi", "stream", "tiles", "deep8")
_UNSET = object()
LOG = logging.getLogger(__name__)


def _tag():
    """a residency tag: the kernel copies the parameter images unless its slot holds this tag"""
    t = (0, 0, 0, 0)
    while not t[0]:
        t = struct.unpack("<4I", secrets.token_bytes(16))
    return t


@dataclass
class CDSPPlan:
    """One compiled model for this plugin's image. framenpu reads ``model``, ``max_crops``,
    ``split`` and the staging factories; the kernel plan itself (descriptor table, window and
    blob placement) is only read back here and by the lease."""

    model: str
    max_crops: int
    slot: int  # residency tag slot and window part, in compile order
    graph: Any  # the lowered model it was compiled from
    plan: Any  # _mercury_layout's plan
    heads: bool  # the heatmap head runs on the DSP
    split: S.Split  # where the graph divides between host and device
    tag: tuple = field(default_factory=_tag)
    crop_ms: float = 0.0  # last measured DSP body per crop: the next request sleeps most of it

    def numpy_staging(self):
        """the layout half for the NumPy host path"""
        return S.NumpyStaging(self.graph, self.plan, self.heads)

    def native_staging(self, layout):
        """the layout half on the layout kernels (CDSPPlugin.layout()); ValueError if this plan
        is not covered"""
        return S.native_staging(layout, self.graph, self.plan, self.heads)


class CDSPPlugin:
    abi = 1  # the ABI this plugin implements: never framenpu's PLUGIN_ABI, which it is checked against
    name = "cdsp"

    def __init__(
        self,
        jobs=None,
        *,
        resident=None,
        axi_mhz=_UNSET,
        hmx_mhz=None,
        _device_factory=Device,
        _lease_factory=arena_lease,
    ):
        """jobs: a qualified Mercury build folder (link.json, image, loader); None: the image
        shipped in framenpu_cdsp/data/mercury. resident: None keeps parameter images in the
        FeatureNN blob region when the image supports it, True requires that, False copies them
        for every crop (legacy plan, same image). axi_mhz: an AXI_RATES rate or None (off); left
        out, FNPU_AXI_MHZ decides. hmx_mhz: an HMX_RATES rate; None: FNPU_HMX_MHZ,
        else 192. A rate either does not know is a ValueError here,
        before any device work. The factories are for tests."""
        if axi_mhz is _UNSET:
            axi_mhz = rate_from_env()
        if axi_mhz is not None and axi_mhz not in AXI_RATES:
            raise ValueError("AXI rate must be one of %s MHz" % sorted(AXI_RATES))
        self.axi_mhz = axi_mhz
        self.hmx_mhz = hmx_rate(hmx_mhz)
        self.jobs = str(BUNDLED_JOBS if jobs is None else jobs)
        self.loader, self.image, f = open_image(self.jobs)
        self.features = f
        if resident and not f.resident:
            raise InvalidModel("this Mercury image has no resident parameter support")
        self.resident = f.resident if resident is None else bool(resident)
        if f.cbias and not self.resident:
            raise InvalidModel("the packed-bias image runs resident requests only")
        # an RS_TILES build keeps 1x1-only tensors in B-tile layout (no gather). A record
        # that misstates it is harmless: a build without RS_TILES ignores the layout word
        # (every tensor stays planar), and one with it runs flag-free rows planar too.
        self.tiles = self.resident and f.tiles
        # 8-block 1x1 groups (--pw-deep --pw-lg 3): flagged ops get a larger scratch. A build
        # without them refuses flag 8; one with them runs unflagged ops 4-block.
        self.deep8 = self.tiles and f.deep8
        self.plans = []
        self._res_base = self._win_base = None
        self._device_factory, self._lease_factory = _device_factory, _lease_factory
        if axi_mhz is not None:
            LOG.info("AXI master raised to %s MHz inside each job", axi_mhz)
        if self.hmx_mhz != DEFAULT_HMX_MHZ:
            LOG.info("HMX clock %s MHz while held", self.hmx_mhz)

    def capabilities(self):
        f = self.features
        return Capabilities(
            abi=self.abi,
            image_sha256=f.sha256,
            max_crops=f.max_crops,
            segments=MULTI_SEGMENTS if f.multi else 1,
            stream=f.stream,
            resident_slots=RES_SLOTS if f.res_slots else int(f.resident),
            features=frozenset(n for n in FEATURE_FLAGS if getattr(f, n))
            | ({"axi"} if self.axi_mhz is not None else set())
            | ({"hmx_fast"} if self.hmx_mhz != DEFAULT_HMX_MHZ else set()),
        )

    def compile(self, graph, batch):
        """Plan a lowered Mercury graph (keypoint, stereo student or detector) for requests of
        1..batch crops, placed after the models compiled before it on this plugin. The first
        model may carry the image's max_crops, later ones up to 3. The DSP runs the
        convolutions (and the heatmap head on a heads image); the hint prefix and the dense
        tails stay on the host."""
        f, slot = self.features, len(self.plans)
        most = f.max_crops if slot == 0 else 3
        if not 1 <= batch <= most:
            return Refusal(f"model {slot}'s requests carry one to {most} crops")
        if slot and not self.resident:
            return Refusal("several models need resident parameters")
        try:
            plan = L.build_plan(
                graph,
                prefix=not graph.detector,
                batch=batch,
                replay=True,
                pw_group=8,
                heads=f.heads and not graph.detector,
                resident=self.resident,
                tiles=self.tiles,
                deep8=self.deep8,
                res_base=self._res_base,
                win_base=self._win_base,
                cbias=f.cbias,
                win_hi=f.win_hi,
                max_crops=f.max_crops,
            )
        except ValueError as exc:
            return Refusal(f"model {slot}: {exc}")
        for k in range(batch):
            for row in struct.iter_unpack("<32I", plan.plan.table):
                if row[0] == L.KIND_RG and f.heads:
                    ok = C.regrid_ok(row)
                else:
                    ok = C.native_ok(
                        C.replay_row(row, k, plan.plan.crop_stride, f.win_hi),
                        s2_direct=0,
                        pw_group=8,
                        mheads=f.heads,
                        resident=f.resident,
                        tiles=self.tiles,
                        deep8=self.deep8,
                        cbias=f.cbias,
                        win_hi=f.win_hi,
                    )
                if not ok:
                    return Refusal("Mercury plan exceeds native bounds")
        self._res_base, self._win_base = plan.res_end, -(-plan.win_end // 4096) * 4096
        heads = f.heads and not graph.detector
        compiled = CDSPPlan(
            getattr(graph, "name", f"model{slot}"),
            batch,
            slot,
            graph,
            plan,
            heads,
            S.split_of(graph, plan, heads),
        )
        self.plans.append(compiled)
        return compiled

    def layout(self):
        """The bundled layout kernels (grid pack/unpack, stem columns) for native staging, or
        None on this platform; opaque to framenpu."""
        return _layout_native.bundled()

    def open(self):
        """The lease on the shared window and image reservation, for the plans compiled so far.
        Each request it runs carries ``sl`` (its plan, or an object exposing the plan's plan,
        slot, tag and crop_ms), ``count`` and ``staged`` (from the plan's staging)."""
        return MercuryLease(
            self.loader,
            self.image,
            self.features,
            self.plans,
            self.resident,
            device_factory=self._device_factory,
            lease_factory=self._lease_factory,
            device_options=self._device_options,
        )

    def _device_options(self):
        """the clock settings for one device session: its own AXI owner (False: none, so the
        device does not read FNPU_AXI_MHZ itself) and HMX power at this plugin's rate"""
        mhz = self.hmx_mhz
        return {
            "_axi": AxiClock(self.axi_mhz) if self.axi_mhz is not None else False,
            "_power_factory": lambda: HmxPower(mhz=mhz),
        }
