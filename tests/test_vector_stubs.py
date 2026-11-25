"""The shipped payloads' exception vectors forward without touching a register.

EVB is shared by every hardware thread. A stub that overwrites a register corrupts whichever
thread the event interrupted: the firmware handler saves registers only from its own first
packet on. Decoded from the payload bytes, so no LLVM and no device."""

from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from framenpu_cdsp.job import Job

BASE, FIRMWARE, OWN = 0xF0A92000, 0x87000000, (2, 4, 6, 8)
DATA = ROOT / "src/framenpu_cdsp/data"


def is_jump(word):
    """J2_jump as a whole packet or its last instruction (parse bits 11)."""
    return word >> 25 == 0b0101100 and word >> 14 & 3 == 3 and not word & 1


def field(word):
    return (word >> 16 & 0x1FF) << 13 | word >> 1 & 0x1FFF


def jump_target(words, index):
    """Target of a one-instruction jump packet at `index`, else None."""
    word = words[index]
    if not is_jump(word):
        return None
    offset = field(word)
    offset -= (offset & 1 << 21) << 1
    return BASE + 4 * index + 4 * offset


def stub_target(words, index):
    """Target of an `immext; jump` packet (no register written) at `index`, else None."""
    ext, word = words[index], words[index + 1]
    if ext >> 28 or ext >> 14 & 3 not in (1, 2) or not is_jump(word):
        return None
    value = ((ext >> 16 & 0xFFF) << 14 | ext & 0x3FFF) << 6 | field(word) & 0x3F
    return BASE + 4 * index + value & 0xFFFFFFFF


def check(code):
    """Raise unless one 256-byte aligned table of 64 jumps forwards transparently."""
    words = struct.unpack(f"<{len(code) // 4}I", code[: len(code) // 4 * 4])
    tables = [
        i
        for i in range(0, len(words) - 63, 64)
        if all(jump_target(words, i + k) is not None for k in range(64))
    ]
    if len(tables) != 1:
        raise ValueError(f"{len(tables)} vector tables")
    own = set()
    for event in range(64):
        target = jump_target(words, tables[0] + event)
        if event in OWN:
            own.add(target)
            continue
        at = target - BASE
        if at % 4 or not 0 <= at <= len(code) - 8:
            raise ValueError(f"event {event} leaves the payload")
        if stub_target(words, at // 4) != FIRMWARE + 4 * event:
            raise ValueError(f"event {event} is not a register-free firmware jump")
    if len(own) != 1:
        raise ValueError("fault events have different handlers")


def payloads():
    return {
        "runtime": (DATA / "runtime.bin").read_bytes(),
        "dispatch_patch": (DATA / "dispatch_patch.bin").read_bytes(),
        "loader": Job.load(DATA / "mercury/teacher_link_loader.fjob").code,
        "vector_add": Job.load(ROOT / "examples/vector_add.fjob").code,
    }


class VectorStubTests(unittest.TestCase):
    def test_every_shipped_payload_forwards_without_a_register(self):
        for name, code in payloads().items():
            with self.subTest(payload=name):
                check(code)

    def test_a_register_stub_and_a_wrong_target_are_rejected(self):
        code = payloads()["dispatch_patch"]
        words = list(struct.unpack(f"<{len(code) // 4}I", code))
        table = next(
            i
            for i in range(0, len(words) - 63, 64)
            if all(jump_target(words, i + k) is not None for k in range(64))
        )
        at = (jump_target(words, table + 18) - BASE) // 4
        old = list(words)
        old[at : at + 2] = 0x08704001, 0x7800C100  # r0 = ##0x87000048
        wrong = list(words)
        wrong[at + 1] += 2 << 1
        for changed in (old, wrong):
            with self.assertRaises(ValueError):
                check(struct.pack(f"<{len(changed)}I", *changed))


if __name__ == "__main__":
    unittest.main()
