import argparse
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread, imshow


def load_standards(standards_dir, bits):
    """ Loads standards and masks from .tif files into numpy arrays """
    standard_imgs = {
        f.name.split("_")[-1].split(".")[0]: imread(f)
        for f in standards_dir.glob(f"standards_{bits}bt_*.tif")
    }
    standard_masks = OrderedDict({
        f.name[:-len("_mask.tif")]: imread(f)
        for f in standards_dir.glob("*_mask.tif")
    })
    return standard_imgs, standard_masks


def construct_standards_df(standard_arrs, mask_arrs):
    # Build individual standards dataframes that house individual element intensities
    dfs = []
    for mineral, mask_arr in mask_arrs.items():
        pixels = []
        for element, img_arr in standard_arrs.items():
            pixels.append(img_arr[mask_arr > 0])

        df = pd.DataFrame(
            np.dstack(list(pixels))[0],
            columns=standard_arrs.keys(),
        )
        df["mineral"] = mineral
        dfs.append(df)

    # Concatenate constituent standards dataframes into one source dataset
    return pd.concat(dfs).reset_index(drop=True)


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
