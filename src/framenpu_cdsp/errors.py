"""The error types of framenpu.errors: the same classes, so framenpu's callers catch them by
either name. They live in framenpu because it must raise and catch them without a backend."""

from framenpu.errors import (
    CompilerError,
    DeviceBusy,
    FrameNPUError,
    InvalidModel,
    JobFailed,
    RecoveryRequired,
    UnsupportedFirmware,
    UnsupportedOperator,
)

__all__ = [
    "FrameNPUError",
    "CompilerError",
    "DeviceBusy",
    "RecoveryRequired",
    "JobFailed",
    "UnsupportedFirmware",
    "InvalidModel",
    "UnsupportedOperator",
]
