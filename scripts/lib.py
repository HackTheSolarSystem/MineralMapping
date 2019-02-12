from collections import OrderedDict

import numpy as np
import pandas as pd
import periodictable
from skimage.io import imread
import yaml

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

def get_standards_weights(standards_dir, minerals):
    if (standards_dir / 'standards.yaml').exists():
        custom = yaml.load((standards_dir / 'standards.yaml').open('r'))
    else:
        custom = {}

    rows = []
    for mineral in minerals:
        if (mineral in custom) and ("formula" not in custom[mineral]):
            weights = custom[mineral]
            weights["mineral"] = mineral
            rows.append(weights)
            continue
        elif (mineral in custom) and ("formula" in custom[mineral]):
            formula = custom[mineral]["formula"]
        else:
            formula = mineral

        weights = {
            str(e): w for e,w
            in periodictable.formula(formula).mass_fraction.items()
        }
        weights["formula"] = formula
        weights["mineral"] = mineral
        rows.append(weights)

    weights_df = pd.DataFrame.from_records(rows).fillna(0)
    return weights_df
