import argparse
import json
from pathlib import Path

from lib import load_standards, construct_standards_df, get_standards_weights


def main(standards_dir, bits=32):
    # Load standards and masks from tif into numpy arrays
    print("Loading standards...")
    standard_arrs, mask_arrs = load_standards(standards_dir, bits)
    print(f"Successfully loaded {len(standard_arrs)} standards with {len(mask_arrs)} masks")

    # Construct the pandas DataFrame containing unmasked intensities of elements along with
    # their corresponding mineral
    df = construct_standards_df(standard_arrs, mask_arrs)
    print()
    print(f"Loaded {len(df)} rows")
    print(f"Mineral counts:\n{json.dumps(df['mineral'].value_counts().to_dict(), indent=4)}")

    weights_df = get_standards_weights(standards_dir, df['mineral'].unique())
    print(weights_df)


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
