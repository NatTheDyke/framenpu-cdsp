import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from framenpu_cdsp import DeviceBusy, Job, RecoveryRequired  # noqa: E402
from framenpu_cdsp.axi import (
    AXI_RATES,
    CFG,
    CMD,
    PLL_PAGE,
    PLL_RESET,
    RCG_PAGE,
    AxiClock,
    rate_from_env,
)  # noqa: E402
from framenpu_cdsp.device import Device  # noqa: E402
from test_library import FakeTransport, Memory  # noqa: E402

CRYSTAL_CFG = 0x00100000  # observed on the Frame during our jobs


class Simulated(AxiClock):
    """Registers with the observed reset values; the PLL locks when enabled in run mode, and
    CMD.UPDATE clears at once (unless the test says otherwise)."""

    def __init__(self, *, lock=True, update=True, cfg=CRYSTAL_CFG, pll=None, events=None):
        self.views = {}
        self.lock, self.update = lock, update
        self.init_cfg, self.init_pll = cfg, dict(pll or PLL_RESET)
        self.events = events
        super().__init__(748.8, mmio_factory=self.make_view)

    def make_view(self, address):
        mm = Memory(4096)
        self.views[address] = mm
        if address == RCG_PAGE:
            struct.pack_into("<I", mm, CFG, self.init_cfg)
        else:
            for off, value in self.init_pll.items():
                struct.pack_into("<I", mm, off, value)
        return mm

    def _w(self, mm, off, value):
        if mm is self.views.get(PLL_PAGE) and off == 0:
            if value & 4 and self._r(mm, 4) == 1 and self.lock:
                value |= 0x80000000
            else:
                value &= ~0x80000000
        if mm is self.views.get(RCG_PAGE) and off == CMD and self.update:
            value &= ~1
        return super()._w(mm, off, value)

    def raise_(self):
        if self.events is not None:
            self.events.append("axi_raise")
        return super().raise_()

    def restore(self):
        if self.events is not None and self.raised:
            self.events.append("axi_restore")
        return super().restore()

    def regs(self):
        r = self._r
        return (
            r(self.views[RCG_PAGE], CMD),
            r(self.views[RCG_PAGE], CFG),
            {o: r(self.views[PLL_PAGE], o) for o in PLL_RESET},
        )


class AxiClockTests(unittest.TestCase):
    def test_raise_selects_core2_and_restore_returns_every_word(self):
        a = Simulated()
        a.raise_()
        cmd, cfg, pll = a.regs()
        self.assertEqual(cfg, CRYSTAL_CFG | 0x503)  # source 5, divide by 2; bit 20 kept
        self.assertEqual(pll[0x10], AXI_RATES[748.8][0])
        self.assertTrue(pll[0] & 0x80000000 and pll[0] & 1)
        a.restore()
        self.assertEqual(a.regs(), (0, CRYSTAL_CFG, PLL_RESET))
        self.assertFalse(a.raised)

    def test_refuses_without_writing_unless_on_the_crystal(self):
        for kwargs in (
            {"cfg": CRYSTAL_CFG | 0x503},
            {"cfg": 0x203},
            {"pll": {**PLL_RESET, 0: 0xE0000005, 0x10: 78}},
        ):
            a = Simulated(**kwargs)
            a._map()
            before = a.regs()
            with self.assertRaises(DeviceBusy):
                a.raise_()
            self.assertEqual(a.regs(), before)
            self.assertFalse(a.raised)

    def test_lock_failure_returns_the_pll_to_reset(self):
        a = Simulated(lock=False)
        with self.assertRaises(RuntimeError):
            a.raise_()
        self.assertEqual(a.regs(), (0, CRYSTAL_CFG, PLL_RESET))
        self.assertFalse(a.raised)

    def test_stuck_update_requires_recovery_and_keeps_the_pll(self):
        a = Simulated(update=False)
        with self.assertRaises(RecoveryRequired):
            a.raise_()
        self.assertTrue(a.abandoned)
        self.assertEqual(a.regs()[2][0x10], 78)  # never stop a PLL the mux may use

    def test_restore_refuses_to_stop_the_pll_if_the_mux_moved(self):
        a = Simulated()
        a.raise_()
        struct.pack_into("<I", a.views[RCG_PAGE], CFG, CRYSTAL_CFG | 0x203)
        with self.assertRaises(RecoveryRequired):
            a.restore()
        self.assertEqual(a.regs()[2][0x10], 78)

    def test_rates_and_environment(self):
        with self.assertRaises(ValueError):
            AxiClock(1500.0)
        old = os.environ.get("FNPU_AXI_MHZ")
        try:
            os.environ.pop("FNPU_AXI_MHZ", None)
            self.assertIsNone(rate_from_env())
            os.environ["FNPU_AXI_MHZ"] = ""
            self.assertIsNone(rate_from_env())
            os.environ["FNPU_AXI_MHZ"] = "748.8"
            self.assertEqual(rate_from_env(), 748.8)
            os.environ["FNPU_AXI_MHZ"] = "1000"
            with self.assertRaises(ValueError):
                rate_from_env()
        finally:
            if old is None:
                os.environ.pop("FNPU_AXI_MHZ", None)
            else:
                os.environ["FNPU_AXI_MHZ"] = old


class DeviceAxiTests(unittest.TestCase):
    def test_raise_after_claim_and_restore_before_ack(self):
        t = FakeTransport()
        a = Simulated(events=t.events)
        with Device(_transport=t, _axi=a) as npu:
            npu.submit(Job(b"\0" * 4), b"input", output_size=4)
        self.assertEqual(t.events, ["claim", "axi_raise", "publish", "axi_restore", "ack", "close"])
        self.assertEqual(a.regs(), (0, CRYSTAL_CFG, PLL_RESET))

    def test_busy_clock_releases_the_gate_unpublished(self):
        t = FakeTransport()
        a = Simulated(cfg=CRYSTAL_CFG | 0x503, events=t.events)
        npu = Device(_transport=t, _axi=a)
        with self.assertRaises(DeviceBusy):
            npu.submit(Job(b"\0" * 4))
        self.assertNotIn("publish", t.events)
        self.assertFalse(t.owned)
        self.assertFalse(t.poisoned)

    def test_missing_completion_leaves_the_clock_raised(self):
        t = FakeTransport(fail=True)
        a = Simulated(events=t.events)
        npu = Device(_transport=t, _axi=a)
        with self.assertRaises(RecoveryRequired):
            npu.submit(Job(b"\0" * 4))
        self.assertTrue(t.poisoned)
        self.assertTrue(a.abandoned)
        self.assertNotIn("axi_restore", t.events)

    def test_default_is_off(self):
        old = os.environ.pop("FNPU_AXI_MHZ", None)
        try:
            self.assertIsNone(Device(_transport=FakeTransport())._axi)
        finally:
            if old is not None:
                os.environ["FNPU_AXI_MHZ"] = old


if __name__ == "__main__":
    unittest.main()
