"""framenpu-cdsp: the Steam Frame CDSP device backend for framenpu.

CDSP transport, firmware validation, power and clocks, VTCM residency, the Mercury kernel ABI
planner, the bundled DSP payloads and the custom-job API. framenpu finds it through the
``framenpu.devices`` entry point; the interface is docs/CONTRACTS.md in the framenpu repository.
"""

from .device import Device
from .errors import (
    CompilerError,
    DeviceBusy,
    FrameNPUError,
    InvalidModel,
    JobFailed,
    RecoveryRequired,
    UnsupportedFirmware,
    UnsupportedOperator,
)
from .job import Job, JobResult
from .model import Clip, Dense, Model, ReLU

__version__ = "0.1.1"

__all__ = [
    "Device",
    "Job",
    "JobResult",
    "Model",
    "Dense",
    "ReLU",
    "Clip",
    "FrameNPUError",
    "CompilerError",
    "DeviceBusy",
    "RecoveryRequired",
    "JobFailed",
    "UnsupportedFirmware",
    "InvalidModel",
    "UnsupportedOperator",
]
