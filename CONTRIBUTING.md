# Contributing

Host-only development works on an ordinary workstation without a headset.

~~~sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ../framenpu -e ".[dev]"
python -m unittest discover -s tests -p 'test_*.py' -v
ruff check .
ruff format --check .
python tools/check_repository.py
~~~

Python 3.11–3.14 are in the configured CI matrix. See docs/VALIDATION.md for
versions actually tested locally.

CI also installs NumPy to exercise the optional numerical tests. To run those
locally, install `numpy>=1.26` in the development environment. Without NumPy,
those tests skip; model-specific tests also need their documented model files.

## Changes

Add tests for behavior, invalid inputs, and meaningful failure paths. Hardware
access must stay lazy: importing, model inspection, tests and packaging cannot
require a headset, root, or /dev/mem. Keep exact firmware guards and the
40 ms host / roughly 50 ms stock completion-window limits. Never automatically
restart tracking services or reclaim possibly active DSP code/data/clocks.

Add kernels through the C ABI before extending the model format. Preserve
serialized format compatibility or explicitly version it. Native jobs are
trusted supervisor code, not sandboxed model data.

## Kernel builds

The runtime builder and custom-job compiler share one implementation.

~~~sh
python tools/build_kernels.py
python tools/build_kernels.py --check
~~~

LLVM 23.1.4 produced the checked-in payloads. Other versions may generate different
code; an untested rebuild must not be described as hardware-validated. Set CLANG,
LD_LLD and LLVM_OBJCOPY to executable paths to select versioned tools.

Regenerate the payload and source-hash manifest after SDK/kernel changes.
The check command verifies generated files without rewriting them.
Explain ABI/kernel changes and provide a sanitized numerical summary when
hardware testing is available. Offline emulation is not hardware proof.

## Hardware and publication

Hardware tests are separate from CI and require --run-on-device. A timeout can
need deliberate DSP/power recovery; closing Python does not reset the engine.

Do not commit credentials, raw machine captures, firmware images, proprietary
weights, or opaque binaries without corresponding source/provenance.

PR descriptions should state the behavior changed, its purpose and checks run,
including hardware limitations. Contributions use the repository's MIT license;
preserve third-party notices.
