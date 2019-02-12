import argparse
import json
from pathlib import Path

from lib import get_standards_characteristics


def main(standards_dir, bits=32):
    characteristics = get_standards_characteristics(standards_dir, bits)
    print(characteristics)





def parse_args():
    """ Build argument parser and get parsed args """

    # Helper function to detect valid directories
    def valid_dir(path_str):
        p = Path(path_str)
        if not p.exists():
            raise argparse.ArgumentTypeError(f"Could not find path {path_str}")
        if not p.is_dir():
            raise argparse.ArgumentTypeError(f"Path {path_str} is not a directory")
        return p

    parser = argparse.ArgumentParser(description="Predict the mineral content of a "
                                                 "meteorite given spectrometer imagery.")
    parser.add_argument("standards_dir", type=valid_dir,
                        help="path to directory containing the standards")
    parser.add_argument("--bits", type=int, choices=[8, 32], default=32,
                        help="image bit-depth to use (8 or 32)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
