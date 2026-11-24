"""Run the compiled C vector-add example on a supported Frame."""

import argparse
from pathlib import Path
import struct

from framenpu_cdsp import Device, Job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=Path(__file__).with_name("vector_add.fjob"))
    args = parser.parse_args()
    job = Job.load(args.job)
    with Device() as npu:
        result = npu.submit(
            job, struct.pack("<8f", 1, 2, 3, 4, 5, 6, 7, 8), output_size=16, params=(4,)
        )
    print(struct.unpack("<4f", result.output))


if __name__ == "__main__":
    main()
