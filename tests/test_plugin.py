"""The cdsp plugin: its registration, the bundled image, capabilities and clock settings."""

import importlib.util
import os
import sys
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from framenpu import devices
from framenpu import contracts
from framenpu.contracts import DEVICE_ENTRY_POINTS

ROOT = Path(__file__).resolve().parents[1]
CDSP = "cdsp=framenpu_cdsp.plugin:CDSPPlugin"


class RegistrationTests(unittest.TestCase):
    def test_the_package_registers_the_cdsp_backend(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        self.assertEqual(
            project["entry-points"][DEVICE_ENTRY_POINTS],
            {"cdsp": "framenpu_cdsp.plugin:CDSPPlugin"},
        )

    def test_the_bundled_image_is_the_qualified_four_crop_build(self):
        import hashlib
        import json

        from framenpu_cdsp import mercury_device as D

        d = D.BUNDLED_JOBS
        self.assertEqual(
            hashlib.sha256((d / "teacher_link.bin").read_bytes()).hexdigest(), D.CROPS4_IMAGE_SHA256
        )
        rec = json.loads((d / "link.json").read_text())
        self.assertEqual(rec["linked"]["sha256"], D.CROPS4_IMAGE_SHA256)
        self.assertEqual(rec["loader"]["sha256"], D.LOADER_SHA256)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required")
    def test_framenpu_serve_uses_the_bundled_image_by_default(self):
        from framenpu import service

        with patch.dict(os.environ, {devices.ENV: CDSP}):
            srv = service.Server("model", None, "unused", os.getuid(), session_factory=lambda: None)
        self.assertEqual((srv.jobs, srv.crops), (None, 4))


class ABITests(unittest.TestCase):
    """The plugin states its own ABI, so a framenpu of another one refuses it."""

    def test_a_framenpu_of_another_abi_refuses_the_plugin(self):
        # as installed beside that framenpu: imported after its PLUGIN_ABI changed
        kept = {n: m for n, m in sys.modules.items() if n.split(".")[0] == "framenpu_cdsp"}
        for n in kept:
            del sys.modules[n]
        try:
            with (
                patch.object(contracts, "PLUGIN_ABI", 2),
                patch.object(devices, "PLUGIN_ABI", 2),
                patch.object(devices, "abi_compatible", lambda abi: abi == 2),
                patch.dict(os.environ, {devices.ENV: CDSP}),
                self.assertRaisesRegex(devices.DeviceUnavailable, "plugin ABI 1; framenpu supp"),
            ):
                devices.load("cdsp")
        finally:
            for n in [n for n in sys.modules if n.split(".")[0] == "framenpu_cdsp"]:
                del sys.modules[n]
            sys.modules.update(kept)

    def test_this_framenpu_accepts_it(self):
        with patch.dict(os.environ, {devices.ENV: CDSP}):
            self.assertTrue(contracts.abi_compatible(devices.load("cdsp").abi))


class CDSPPluginTests(unittest.TestCase):
    def test_bundled_image_capabilities(self):
        with patch.dict(os.environ, {devices.ENV: CDSP}):
            plugin = devices.load("cdsp")
        from framenpu_cdsp import mercury_device as D

        caps = plugin.capabilities()
        self.assertEqual(plugin.jobs, str(D.BUNDLED_JOBS))
        self.assertEqual(caps.image_sha256, D.CROPS4_IMAGE_SHA256)
        self.assertEqual((caps.abi, caps.max_crops, caps.segments, caps.stream), (1, 4, 2, True))
        self.assertEqual(caps.resident_slots, D.RES_SLOTS)
        self.assertLessEqual({"heads", "resident", "cbias", "tiles", "deep8"}, caps.features)

    def test_compile_refuses_requests_the_image_cannot_carry(self):
        from framenpu.contracts import Refusal
        from framenpu_cdsp.plugin import CDSPPlugin

        plugin = CDSPPlugin()
        for batch in (0, 5):
            got = plugin.compile(object(), batch)  # refused before the graph is read
            self.assertIsInstance(got, Refusal)
            self.assertEqual(got.reason, "model 0's requests carry one to 4 crops")
        self.assertEqual(plugin.plans, [])

    def test_an_unqualified_folder_is_refused(self):
        from framenpu_cdsp.plugin import CDSPPlugin

        with self.assertRaises(Exception):
            CDSPPlugin(jobs=ROOT / "tests")


class _Opened(Exception):
    pass


class ClockSettingTests(unittest.TestCase):
    """The job-scoped AXI raise and the HMX rate are plugin settings: from the option or the
    environment, refused at load when the rate is unknown, reported as features, and handed to
    each device session by the lease."""

    def plugin(self, env=None, hmx_env=None, **options):
        from framenpu_cdsp.plugin import CDSPPlugin

        with patch.dict(os.environ):
            for name, value in (("FNPU_AXI_MHZ", env), ("FNPU_HMX_MHZ", hmx_env)):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            return CDSPPlugin(**options)

    def axi(self, p):
        return p._device_options()["_axi"]

    def test_off_by_default(self):
        p = self.plugin()
        self.assertIsNone(p.axi_mhz)
        self.assertEqual(p.hmx_mhz, 192.0)
        self.assertFalse({"axi", "hmx_fast"} & p.capabilities().features)
        self.assertIs(self.axi(p), False)  # the device does not read FNPU_AXI_MHZ itself
        self.assertEqual(p._device_options()["_power_factory"]().mhz, 192.0)

    def test_the_option_and_the_environment_turn_it_on(self):
        from framenpu_cdsp.axi import AxiClock

        for p in (self.plugin(axi_mhz=748.8), self.plugin("748.8")):
            self.assertEqual(p.axi_mhz, 748.8)
            self.assertIn("axi", p.capabilities().features)
            a, b = self.axi(p), self.axi(p)
            self.assertIsInstance(a, AxiClock)
            self.assertIsNot(a, b)  # one owner per device session
            self.assertEqual(a.mhz, 748.8)
        self.assertIsNone(self.plugin("748.8", axi_mhz=None).axi_mhz)  # the option wins

    def test_an_unknown_rate_is_refused_at_load(self):
        for kw in (
            {"env": "1500"},
            {"env": "fast"},
            {"axi_mhz": 500.0},
            {"hmx_env": "672"},
            {"hmx_mhz": 806.4},
        ):
            with self.assertRaises(ValueError):
                self.plugin(**kw)

    def test_the_option_and_the_environment_raise_the_hmx_rate(self):
        for p in (self.plugin(hmx_mhz=499.2), self.plugin(hmx_env="499.2")):
            self.assertEqual(p.hmx_mhz, 499.2)
            self.assertIn("hmx_fast", p.capabilities().features)
            # the session's power owner runs at the plugin's rate, whatever the env says later
            with patch.dict(os.environ, {"FNPU_HMX_MHZ": "192"}):
                self.assertEqual(p._device_options()["_power_factory"]().mhz, 499.2)
        self.assertEqual(self.plugin(hmx_env="499.2", hmx_mhz=192.0).hmx_mhz, 192.0)

    def test_the_lease_gives_each_device_session_the_plugins_owner(self):
        from unittest.mock import MagicMock

        seen = []

        def device(**kw):
            seen.append(kw)
            raise _Opened

        for mhz in (748.8, None):
            p = self.plugin(axi_mhz=mhz, _device_factory=device, _lease_factory=MagicMock)
            with self.assertRaises(_Opened):
                p.open().open()
        self.assertEqual([k["dispatch"] for k in seen], ["direct", "direct"])
        self.assertEqual(seen[0]["_axi"].mhz, 748.8)
        self.assertIs(seen[1]["_axi"], False)
        self.assertEqual([k["_power_factory"]().mhz for k in seen], [192.0, 192.0])


if __name__ == "__main__":
    unittest.main()
