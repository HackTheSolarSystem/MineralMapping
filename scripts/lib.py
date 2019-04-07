from collections import OrderedDict
import json
import logging

import numpy as np
import pandas as pd
import periodictable
from skimage.io import imread
from sklearn.linear_model import LinearRegression
import yaml


def load_images(directory, bits, mask=None):
    """ Loads meteorite sample images from .tif files into numpy arrays """
    elements = []
    pixels = []
    shape = None
    for path in directory.glob(f"*_{bits}bt_*.tif"):
        #import pdb; pdb.set_trace()
        #print(path.basename)
        elements.append(path.stem.split("_")[-1])
        if shape is None:
            shape = imread(path).shape
        pixels.append(imread(path).flatten())

    df = pd.DataFrame(np.dstack(pixels)[0], columns=elements)
    df = df.reset_index().rename(columns={"index": "order"})

    if mask:
        df['mask'] = (imread(mask).flatten() > 0).astype(int)
    else:
        df['mask'] = 1

    return df, shape



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
    """ Constructs a DataFrame containing all unmasked pixels in given standards """
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


def load_standards_df(standards_dir, bits):
    """ Load a dataframe containing all unmasked points in standards """
    # Load standards and masks from tif into numpy arrays
    #print("Loading standards...")
    standard_arrs, mask_arrs = load_standards(standards_dir, bits)
    #print(f"Successfully loaded {len(standard_arrs)} standards with {len(mask_arrs)} masks")

    # Construct the pandas DataFrame containing unmasked intensities of elements along with
    # their corresponding mineral
    df = construct_standards_df(standard_arrs, mask_arrs)
    #print()
    #print(f"Loaded {len(df)} rows")
    #print(f"Mineral counts:\n{json.dumps(df['mineral'].value_counts().to_dict(), indent=4)}")
    return df


def get_formula(formula_str, format="fraction"):
    """
        Convert a chemical formula string to a dictionary of element weights.
        If format is "fraction" then return the fraction of the total weight
        of the formula that that element comprises.
        If format is "mass" then return the absolute mass of that element
        in the molecule.

    """
    try:
        formula = periodictable.formula(formula_str)
        multiplier = 1 if format == "fraction" else formula.mass
        return {
            str(element): weight*multiplier for element, weight
            in formula.mass_fraction.items()
        }
    except Exception:
        return None


def get_standards_weights(standards_dir, minerals):
    """ Builds a DataFrame containing theoretical weight proportions for given minerals """
    # Load custom weight proportion overrides
    custom = {}
    if (standards_dir / 'standards.yaml').exists():
        custom = yaml.load((standards_dir / 'standards.yaml').open('r'))

    rows = []
    for mineral in minerals:
        if mineral not in custom:
            # Assume the mineral name is its chemical formula
            formula_str = mineral
            weights = get_formula(formula_str)
        elif isinstance(custom[mineral], str):
            # Get the formula from the overrides
            formula_str = custom[mineral]
            weights = get_formula(formula_str)
        else:
            # Pull the weights directly from the overrides
            formula_str = None
            weights = custom[mineral]

            for e, v in weights.items():
                if v > 1:
                    raise ValueError(
                        f"{mineral}[{e}] must be a decimal less than 1, not a percent."
                        f" Got {v}. Did you mean {v/100}? "
                    )

        if weights is None:
            print(f"Invalid formula for mineral {mineral}: {formula_str}. Skipping")
            continue

        weights = {
            f"{element}_weight": weight
            for element, weight in weights.items()
        }
        weights["mineral"] = mineral
        weights["formula"] = formula_str
        rows.append(weights)

    # Build the dataframe
    return pd.DataFrame.from_records(rows).fillna(0)


def calculate_element_characteristics(df, elements):
    """ Contsruct a linear regression that fits the elements' intensities
        to their calculated theoretical weights
    """
    results = {}
    for col in elements:
        if f"{col}_weight" not in df.columns:
            continue

        x = df['%s_weight' % col].values.reshape(-1,1)
        y = df[col]

        # Should fit_intercept be True? The assumption is that it passes
        # through the origin
        model = LinearRegression(fit_intercept=False)
        model.fit(x,y)

        d = {
            'element': col,
            'coef': model.coef_[0],
            'intercept': model.intercept_,
            # TODO Handle this hack with ignoring low weights?
            'std': df[
                df['%s_weight' % col] > .01
            ].groupby('mineral')[col].std().mean(),
            'noise': df[df['%s_weight' % col] == 0][col].std()
        }
        results[col] = d

        if d['std'] > 20:
            logging.warning(f"{col} std > 10 ({d['std']})")
        if d['noise'] > 5:
            logging.warning(f"{col} noise > 5 ({d['noise']})")
    return results


def get_standards_characteristics(standards_dir, bits=32, manual_elements=True):
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
    # Load pixels from standards
    df = load_standards_df(standards_dir, bits)
    elements = [ v for v in df.columns.values if v != "mineral" ]

    # Load the expected mineral weights and perform a regression to get
    # the mapping.
    weights_df = get_standards_weights(standards_dir, df['mineral'].unique())
    df = df.merge(weights_df, on='mineral')

    def mineral_diagnostics(group):
        mineral = group['mineral'].iloc[0]

        for element in elements:
            weight = f"{element}_weight"
            if weight not in group.columns:
                continue

            #print(element, group[weight].mean(), group[element].mean(), group[element].std())

            if (group[weight].mean() < .01) and (group[element].mean() > 10):
                logging.warning(
                    f"{mineral} {element} channel values unexpectedly high"
                    f" (mean = {group[element].mean()})"
                )
            if group[element].std() > 20:
                logging.warning(
                    f"{mineral} {element} channel STD > 20 ({group[element].std()})"
                )

    df.groupby('mineral').apply(mineral_diagnostics)

    elements = calculate_element_characteristics(df, elements)

    # Include manual element characteristics
    if manual_elements and (standards_dir / 'elements.yaml').exists():
        elements.update(yaml.load((standards_dir / 'elements.yaml').open('r')))

    return elements

def load_target_minerals(target_path):
    return yaml.load(target_path.open('r'))
