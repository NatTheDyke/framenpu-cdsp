"""Create a portable synthetic model; no hardware access."""

import argparse
from pathlib import Path

from framenpu_cdsp import Dense, Model

DEFAULT = Path(__file__).with_name("xor.fnpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Model(2, [Dense([[1, -1], [-1, 1]], activation="relu"), Dense([[1], [1]])], name="xor").save(
        args.output
    )
    print(args.output)


if __name__ == "__main__":
    main()
