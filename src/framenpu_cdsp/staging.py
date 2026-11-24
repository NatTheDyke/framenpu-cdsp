"""The layout half of a Mercury request: the host's FP16 tensors -> the kernel's staged grid words
in the shared window, and the device's grids -> FP32 maps (framenpu docs/CONTRACTS.md).

framenpu (the model half) never sees the kernel's layout. A compiled plan tells it, through its
``Split``, which graph tensors the host stages and which come back; framenpu writes FP16 values
into a staging workspace (native path) or hands over FP32 arrays (NumPy path), and reads the
returned maps by graph tensor name. The padded heatmap (21 channels + 1 zero channel) comes back
as its ``heatmap_xy`` view.

Native workspaces are the layout half of what framenpu._mercury_host binds once per batch size:
the same buffers, pointers and layout-kernel calls in the same order as before the split, so a
request costs no extra kernel work.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import _mercury_common as C
from . import _mercury_layout as L


@dataclass(frozen=True)
class Split:
    """Where a compiled model divides between host and device, in graph terms."""

    prefix: tuple  # ops the host runs (in order) to produce staged tensors from the inputs
    staged: tuple  # graph tensors the host stages per crop: inputImg, then the hint if any
    last: str  # the last op of the device part
    skip: tuple  # ops after ``last`` the device also runs (the heatmap head on a heads image)
    returned: tuple  # graph tensors the device returns per crop, as FP32 maps


def split_of(graph, mplan, heads):
    if graph.detector:
        return Split((), ("inputImg",), mplan.ops[-1].name, (), (mplan.final,))
    return Split(
        tuple(L.HOST_HINT_OPS[:3]),
        ("inputImg", L.HINT),
        L.LAST,
        tuple(L.HEAD_OPS) if heads else (),
        (L.OUTPUT, "heatmap_xy") if heads else (L.OUTPUT,),
    )


def _returned(maps):
    """the device maps by graph tensor name: the padded heatmap as its heatmap_xy view"""
    if L.HEATMAP in maps:
        return {L.OUTPUT: maps[L.OUTPUT], "heatmap_xy": maps[L.HEATMAP][:21]}
    return maps


# ---------------------------------------------------------------- NumPy path
class NumpyStaging:
    """Layout for the NumPy host path: request(prepared) converts each crop's FP32 tensors to the
    kernel's layout tensors (stem columns, hint) at prepare time, outside the gate."""

    def __init__(self, graph, mplan, heads):
        self.mplan, self.detector = mplan, graph.detector
        self.read_names = (
            (mplan.final,) if graph.detector else (L.OUTPUT, L.HEATMAP) if heads else (L.OUTPUT,)
        )

    def request(self, prepared):
        return _NumpyRequest(self, prepared)


class _NumpyRequest:
    def __init__(self, st, prepared):
        self.st = st
        self.tensors = []
        for t in prepared:
            img = t["inputImg"]
            ts = {L.STEM_COLS: L.stem_columns(img, img.shape[-1])}
            if not st.detector:
                ts[L.HINT] = t[L.HINT]
            self.tensors.append(ts)
        self.maps = [None] * len(prepared)

    def stage(self, mem):
        for k, ts in enumerate(self.tensors):
            L.stage_inputs(
                self.st.mplan.crop(k),
                lambda at, data: mem.__setitem__(slice(at, at + len(data)), data),
                ts,
            )

    def read(self, get, k):
        """Read and check crop k's maps (get(at, n) reads the window); False if invalid."""
        import numpy as np

        maps = {}
        for name in self.st.read_names:
            out, stray = L.read_tensor(self.st.mplan.crop(k), get, name)
            if stray or not np.isfinite(out).all():
                return False
            maps[name] = out.astype(np.float32)
        self.maps[k] = _returned(maps)
        return True


# ---------------------------------------------------------------- native path
def _ptr(a):
    if not a.flags.c_contiguous:
        raise ValueError("native workspace arrays must be C-contiguous")
    return a.ctypes.data


class NativeStaging:
    """Layout half of a keypoint or stereo-student request on the layout kernels; refuses any
    plan it does not cover (the session then uses NumPy)."""

    def __init__(self, layout, graph, mplan, heads):
        import numpy as np

        if layout is None:
            raise ValueError("the layout kernels are not available")
        self.layout, self.heads, self.mplan = layout, heads, mplan
        self.views = getattr(graph, "views", 1)
        mp = mplan.crop(0)
        if list(mp.external + mp.host_parts) != [L.STEM_COLS, L.HINT]:
            raise ValueError("native staging covers the stem columns and the hint plane only")
        self.grids = {}
        names = [L.STEM_COLS, L.HINT, L.OUTPUT] + ([L.HEATMAP] if heads else [])
        for name in names:
            g, c = mp.grid(name), mp.shape[name][0]
            idx, _ = C._grid_index(g.w, g.h)
            nwords = g.nbin * 32
            if c % 2 or g.tensor_bytes(c) != 2 * c * nwords:
                raise ValueError(f"unexpected grid for {name}")
            for k in range(len(mplan.crops) or 1):
                ck = mplan.crop(k)
                if ck.grid(name) != g or ck.shape[name] != mp.shape[name]:
                    raise ValueError(f"crop {k} lays out {name} differently")
            self.grids[name] = (c, np.ascontiguousarray(idx, np.int32), nwords, g)
        if self.grids[L.STEM_COLS][0] != L.stem_planes(self.views) or self.grids[L.HINT][0] != 32:
            raise ValueError("unexpected staged channel counts")

    def work(self, nb):
        return _StageWork(self, nb)


class _StageWork:
    """Layout buffers of one nb-crop workspace. The model half writes each crop's FP16 image to
    img16 and calls pack_image(k), and the hint layer writes hint16; pack_hint() packs it."""

    def __init__(self, ns, nb):
        import numpy as np

        self.ns, self.nb = ns, nb
        self.lib = ns.layout.lib
        u16, f32 = np.uint16, np.float32
        self.views = ns.views
        self.img16 = np.zeros(ns.views * 128 * 128, u16)
        self.cols = np.zeros((9 * ns.views + 1) * 4096, u16)  # + the plane the last call zeroes
        self.img16_p, self.cols_p = _ptr(self.img16), _ptr(self.cols)
        c, idx, nwords, _ = ns.grids[L.STEM_COLS]
        self.stem = np.zeros((nb, c * nwords), u16)
        self.stem_pack = [
            (self.cols_p, c, idx.size, _ptr(idx), nwords, _ptr(self.stem[k])) for k in range(nb)
        ]
        self.hint16 = np.zeros((nb, 8192), u16)
        self.hint16_p = _ptr(self.hint16)
        c, idx, nwords, _ = ns.grids[L.HINT]
        self.hint_grid = np.zeros((nb, c * nwords), u16)
        self.hint_pack = [
            (_ptr(self.hint16[k]), c, idx.size, _ptr(idx), nwords, _ptr(self.hint_grid[k]))
            for k in range(nb)
        ]
        self.staged = [
            (
                (L.STEM_COLS, memoryview(self.stem[k]).cast("B")),
                (L.HINT, memoryview(self.hint_grid[k]).cast("B")),
            )
            for k in range(nb)
        ]
        self.read_names = (L.OUTPUT, L.HEATMAP) if ns.heads else (L.OUTPUT,)
        self.stats = np.zeros(2, np.int32)
        self.raw, self.raw_bytes, self.maps, self.unpack = {}, {}, {}, {}
        for name in self.read_names:
            c, idx, nwords, g = ns.grids[name]
            self.raw[name] = np.zeros(c * nwords, u16)
            self.raw_bytes[name] = memoryview(self.raw[name]).cast("B")
            self.maps[name] = np.zeros((nb, c, g.h, g.w), f32)
            self.unpack[name] = [
                (
                    _ptr(self.raw[name]),
                    c,
                    idx.size,
                    _ptr(idx),
                    nwords,
                    _ptr(self.maps[name][k]),
                    _ptr(self.stats),
                )
                for k in range(nb)
            ]
        # the returned tensors by graph name: (nb, ...) FP32 arrays the model half's tail reads
        self.outputs = {L.OUTPUT: self.maps[L.OUTPUT]}
        if ns.heads:
            self.outputs["heatmap_xy"] = self.maps[L.HEATMAP][:, :21]

    def pack_image(self, k):
        """crop k's FP16 image (img16) -> its staged stem columns"""
        lay = self.lib
        for c in range(self.views):  # 9 planes per view, each call zeroes the next plane
            lay.fnpu_stem_columns(self.img16_p + c * 2 * 128 * 128, self.cols_p + c * 2 * 9 * 4096)
        lay.fnpu_grid_pack(*self.stem_pack[k])

    def pack_hint(self):
        """every crop's FP16 hint plane (hint16) -> its staged grid words"""
        for args in self.hint_pack:
            self.lib.fnpu_grid_pack(*args)

    def stage(self, mem):
        for k in range(self.nb):
            where = self.ns.mplan.crop(k).where
            for name, words in self.staged[k]:
                at = where[name]
                mem[at : at + len(words)] = words

    def read(self, get, k):
        """Unpack and check crop k's device outputs into the maps; False if invalid."""
        lib = self.lib
        where = self.ns.mplan.crop(k).where
        for name in self.read_names:
            raw = self.raw[name]
            self.raw_bytes[name][:] = get(where[name], raw.nbytes)
            lib.fnpu_grid_unpack(*self.unpack[name][k])
            if self.stats[0] or self.stats[1]:
                return False
        return True

    def load_maps(self, maps_list):
        """Put given device maps (by kernel tensor name) in the workspace (tests)."""
        for k, maps in enumerate(maps_list):
            for name in self.read_names:
                self.maps[name][k] = maps[name]


class NativeDetStaging:
    """Layout half of a hand-detector request: the stem im2col and grid packing as one
    precomputed gather (every staged half is an image pixel or zero), the readback by
    fnpu_grid_unpack. Refuses any other plan."""

    def __init__(self, layout, graph, mplan, size):
        import numpy as np

        if layout is None:
            raise ValueError("the layout kernels are not available")
        self.layout, self.mplan = layout, mplan
        mp = mplan.crop(0)
        if list(mp.external + mp.host_parts) != [L.STEM_COLS] or mplan.final != L.DET_OUTPUT:
            raise ValueError(
                "native detector staging covers the stem columns of a whole device part"
            )
        # staged stem columns: half (pair p, word i, j) holds plane 2p+j at the position
        # whose word is i (C._grid_index), plane 3dy+dx = image[2y-1+dy, 2x-1+dx]; anything
        # else (outside the image, plane 9, non-interior words) is the zero at index size^2
        c = mp.shape[L.STEM_COLS][0]
        g = mp.grid(L.STEM_COLS)
        h = size // 2
        if c != L.stem_planes(1) or (g.w, g.h) != (h, h):
            raise ValueError("unexpected stem grid")
        idx, _ = C._grid_index(g.w, g.h)
        nwords = g.nbin * 32
        if g.tensor_bytes(c) != 2 * c * nwords:
            raise ValueError("unexpected stem grid size")
        zero = size * size
        gather = np.full((c // 2, nwords, 2), zero, np.int32)
        yy, xx = np.divmod(np.arange(h * h), h)
        for plane in range(9):
            dy, dx = divmod(plane, 3)
            iy, ix = 2 * yy - 1 + dy, 2 * xx - 1 + dx
            src = np.where((iy >= 0) & (iy < size) & (ix >= 0) & (ix < size), iy * size + ix, zero)
            gather[plane // 2, idx, plane % 2] = src
        self.gather = gather.reshape(-1)
        self.size = size

        gf = mp.grid(L.DET_OUTPUT)
        cf = mp.shape[L.DET_OUTPUT][0]
        fidx, _ = C._grid_index(gf.w, gf.h)
        fwords = gf.nbin * 32
        if cf % 2 or gf.tensor_bytes(cf) != 2 * cf * fwords or cf * gf.w * gf.h != 1440:
            raise ValueError("unexpected final grid")
        self.final = (cf, np.ascontiguousarray(fidx, np.int32), fwords)
        for k in range(len(mplan.crops) or 1):
            ck = mplan.crop(k)
            for name in (L.STEM_COLS, L.DET_OUTPUT):
                if ck.grid(name) != mp.grid(name) or ck.shape[name] != mp.shape[name]:
                    raise ValueError(f"crop {k} lays out {name} differently")

    def work(self, nb):
        return _DetStageWork(self, nb)


class _DetStageWork:
    def __init__(self, nd, nb):
        import numpy as np

        self.nd, self.nb = nd, nb
        n = nd.size * nd.size
        self.img16 = np.zeros(n + 1, np.uint16)  # + the zero every outside half reads
        self.img16_p = _ptr(self.img16)
        self.stem = np.zeros((nb, nd.gather.size), np.uint16)
        self.staged = [memoryview(self.stem[k]).cast("B") for k in range(nb)]
        c, idx, nwords = nd.final
        self.raw = np.zeros(c * nwords, np.uint16)
        self.raw_bytes = memoryview(self.raw).cast("B")
        self.stats = np.zeros(2, np.int32)
        self.maps = np.zeros((nb, c * idx.size), np.float32)  # flattened = Flatten_163
        self.unpack = [
            (_ptr(self.raw), c, idx.size, _ptr(idx), nwords, _ptr(self.maps[k]), _ptr(self.stats))
            for k in range(nb)
        ]
        self.outputs = {L.DET_OUTPUT: self.maps}

    def pack_image(self, k):
        import numpy as np

        np.take(self.img16, self.nd.gather, out=self.stem[k])

    def stage(self, mem):
        for k in range(self.nb):
            at = self.nd.mplan.crop(k).where[L.STEM_COLS]
            mem[at : at + len(self.staged[k])] = self.staged[k]

    def read(self, get, k):
        """Unpack and check crop k's final map; False if invalid."""
        at = self.nd.mplan.crop(k).where[L.DET_OUTPUT]
        self.raw_bytes[:] = get(at, self.raw.nbytes)
        self.nd.layout.lib.fnpu_grid_unpack(*self.unpack[k])
        return not (self.stats[0] or self.stats[1])

    def load_maps(self, maps_list):
        """Put given device maps in the workspace (tests)."""
        for k, maps in enumerate(maps_list):
            self.maps[k] = maps[L.DET_OUTPUT].reshape(self.maps[k].shape)


def native_staging(layout, graph, mplan, heads):
    """The native layout half for this plan, or ValueError when the plan is not covered."""
    if graph.detector:
        return NativeDetStaging(layout, graph, mplan, graph.inputs["inputImg"][-1])
    return NativeStaging(layout, graph, mplan, heads)
