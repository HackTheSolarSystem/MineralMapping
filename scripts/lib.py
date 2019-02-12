from collections import OrderedDict
import json

import numpy as np
import pandas as pd
import periodictable
from skimage.io import imread
from sklearn.linear_model import LinearRegression
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
    weights_df.columns = [str(i) + '_weight' if str(i)[0].isupper() else str(i) for i in weights_df.columns]
    return weights_df

def calculate_element_characteristics(df, elements):
    results = {}
    for col in elements:
        if f"{col}_weight" in df.columns:
            x = df['%s_weight' % col].values.reshape(-1,1)
            y = df[col]

            model = LinearRegression()
            model.fit(x,y)

            d = {
                'element': col,
                'coef': model.coef_[0],
                'intercept': model.intercept_,
                # TODO Handle this hack with ignoring low weights?
                'std': df[df['%s_weight' % col] > .01][col].std(),
                'noise': df[df['%s_weight' % col] == 0][col].std()
            }
            results[col] = d
    return results

def get_standards_characteristics(standards_dir, bits=32):
    """
    Given a standards directory following a specified format, return a
    dictionary describing the mapping between the spectrometer intensities
    and an elements weight percentage in a mineral along with a description
    of the distribution.

    The return format is a dictionary of elements:dictionary in the following
    format:

        {
            'element': element name,
            'coef': a number to multiply the elements weight percentage to get
                the expected intensity.
            'intercept': The y intercept, should be near 0.
            'std': The standard distribution of the intensity readings
            'noise': The estimated noise in the readings for that element.
        }
    """
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

    # Load the expected mineral weights and perform a regression to get
    # the mapping.
    weights_df = get_standards_weights(standards_dir, df['mineral'].unique())
    df = df.merge(weights_df, on='mineral')
    elements = calculate_element_characteristics(df, standard_arrs.keys())

    return elements
