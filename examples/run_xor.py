"""Run the synthetic model on a supported Frame. HMX setup requires root."""

import argparse
from pathlib import Path

from framenpu_cdsp import Device, Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("xor.fnpu"))
    args = parser.parse_args()
    model = Model.load(args.model)
    with Device() as npu:
        result = model.run(npu, [[0, 0], [0, 1], [1, 0], [1, 1]])
    print(result)  # [[0.0], [1.0], [1.0], [0.0]]


if __name__ == "__main__":
    main()
