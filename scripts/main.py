import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread

parser = argparse.ArgumentParser(description='Predict the mineral content of a meteorite given spectrometer imagary.')
parser.add_argument('standards', type=str, help='a directory containing the standards.')
parser.add_argument('--bits', default='32', help='image bit-depth to use (8 or 32)')

args = parser.parse_args()
print(args)

def load_standards_df(args):
    standard_path = Path(args.standards)

    # Get the mask paths
    standard_masks = [
        i for i in list(standard_path.glob('*_mask.tif')) if 'obj' not in i.name
    ]

    # Load the individual element images for the standards
    standard_elements = [
        {'name': s.name.split('_')[2].split('.')[0], 'image': imread(s)}
        for s in standard_path.glob('standards_%s*.tif' % args.bits)
    ]

    dfs = []
    for s in standard_masks:
        mask = imread(s)
        pixels = []
        for element in standard_elements:
            # Get the pixels in the mask location for each element
            pixels.append(element['image'][mask > 0])

        # Create a dataframe of the pixels
        df = pd.DataFrame(
            np.dstack(pixels)[0], columns=[i['name'] for i in standard_elements]
        )
        df['mineral'] = s.stem.split('_')[0]
        dfs.append(df)

    # Concatenate all of the dataframes
    df = pd.concat(dfs).reset_index(drop=True)

    return df

df = load_standards_df(args)
print(df.head())
