#!/usr/bin/env python3
"""Build the FrameNPU layout kernels as a freestanding shared library for ctypes.

  layout  src/framenpu_cdsp/host/fnpu_layout.c  window reads, stem columns, grid pack/unpack
  host    the installed framenpu's host/fnpu_host.c, for local test builds only (framenpu bundles it)

    python tools/build_host_native.py                              # bundled aarch64 + manifest
    python tools/build_host_native.py --target x86_64 --output DIR   # local test build

The bundled build writes src/framenpu_cdsp/data/layout_aarch64.so and layout_aarch64.json (bytes,
sha256, source sha256, flags, toolchain). The library has no libc or Python dependency, so
one build serves every Python on the Frame. Needs clang and ld.lld.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import framenpu

ROOT = Path(__file__).resolve().parents[1]
BASE_FLAGS = (
    "-O2",
    "-std=c11",
    "-fPIC",
    "-shared",
    "-nostdlib",
    "-ffreestanding",
    "-ffp-contract=off",
    "-fno-stack-protector",
    "-fno-unwind-tables",
    "-fno-asynchronous-unwind-tables",
    "-fvisibility=hidden",
    "-Wall",
    "-Werror",
    "-fuse-ld=lld",
    "-Wl,--build-id=none",
    "-Wl,-z,now",
    "-Wl,-z,relro",
    "-Wl,-z,noexecstack",
    "-Wl,--no-undefined",
)
LIBS = {
    "host": {
        "package": Path(framenpu.__file__).resolve().parent,
        "source": "host/fnpu_host.c",
        "format": "framenpu-host-native",
        "abi_symbol": "fnpu_abi",
        "soname": "libfnpu_host.so",
        "exports": (
            "fnpu_abi",
            "fnpu_nonfinite32",
            "fnpu_to_f16",
            "fnpu_from_f16",
            "fnpu_dense",
            "fnpu_conv1x1",
            "fnpu_avgpool",
            "fnpu_hint",
            "fnpu_conv1x1_f16",
        ),
    },
    "layout": {
        "package": ROOT / "src/framenpu_cdsp",
        "source": "host/fnpu_layout.c",
        "format": "framenpu-cdsp-layout-native",
        "abi_symbol": "fnpu_layout_abi",
        "soname": "libfnpu_layout.so",
        "exports": (
            "fnpu_layout_abi",
            "fnpu_read32",
            "fnpu_stem_columns",
            "fnpu_grid_pack",
            "fnpu_grid_unpack",
        ),
    },
}
COMMON = "host/fnpu_common.h"


def flags(lib):
    return BASE_FLAGS + (f"-Wl,-soname,{LIBS[lib]['soname']}",)


def _tool(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def build(target, out, lib="host"):
    spec = LIBS[lib]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = spec["package"] / spec["source"]
    _tool(["clang", f"--target={target}-linux-gnu", *flags(lib), "-o", str(out), str(src)])
    syms = _tool(["llvm-nm", "-D", "--defined-only", str(out)]).split()
    undefined = _tool(["llvm-nm", "-D", "--undefined-only", str(out)]).split()
    if undefined:
        raise SystemExit(f"library imports symbols: {undefined}")
    missing = [e for e in spec["exports"] if e not in syms]
    if missing:
        raise SystemExit(f"library lacks exports: {missing}")
    data = out.read_bytes()
    abi = re.search(rf"{spec['abi_symbol']}\(void\) \{{ return (\d+); \}}", src.read_text())
    return {
        "format": spec["format"],
        "abi": int(abi[1]),
        "target": f"{target}-linux-gnu",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((spec["package"] / name).read_bytes()).hexdigest()
            for name in (spec["source"], COMMON)
        },
        "compiler_flags": list(flags(lib)),
        "exports": list(spec["exports"]),
        "toolchain": {
            "clang": _tool(["clang", "--version"]).splitlines()[0].strip(),
            "linker": _tool(["ld.lld", "--version"]).strip(),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default="layout", choices=("host", "layout"))
    ap.add_argument("--target", default="aarch64", choices=("aarch64", "x86_64"))
    ap.add_argument("--output", help="directory for a local (non-bundled) build")
    a = ap.parse_args()
    for lib in (a.lib,):
        if a.output is None:
            if a.target != "aarch64" or lib != "layout":
                raise SystemExit("only the aarch64 layout library is bundled; pass --output")
            data = LIBS[lib]["package"] / "data"
        else:
            data = Path(a.output)
        path = data / f"{lib}_{a.target}.so"
        record = build(a.target, path, lib)
        (data / f"{lib}_{a.target}.json").write_text(json.dumps(record, indent=2) + "\n")
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(
            json.dumps(
                {"library": str(shown), "bytes": record["bytes"], "sha256": record["sha256"]}
            )
        )


if __name__ == "__main__":
    sys.exit(main())
