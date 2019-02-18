import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib import get_standards_characteristics, load_target_minerals, get_formula

def get_variable_percent(formula, n, epsilon=.0001):
    elements = [
        {'element': e, 'min': m[0], 'max': m[1]}
        for e, m in formula.items()
    ]

    base = sum([e['min'] for e in elements])
    remainder = 1 - base
    element_remainders = [e['max'] - e['min'] for e in elements]

    v = np.hstack([
        np.random.uniform(0, e, (n, 1)) for e in element_remainders
    ])

    while remainder > 0:
        s = v.sum(axis=1, keepdims=True)
        v = (v/s)*remainder
        mask = v < element_remainders
        r = np.clip(v - element_remainders, 0, None)
        v = v - r
        v = v + (mask * (r.sum(axis=1) / mask.sum(axis=1)).reshape(-1, 1))
        if np.abs(remainder - v.sum(axis=1)).mean() < epsilon:
            break

    return [(e['element'], e['min']+v[:, i]) for i, e in enumerate(elements)]

def simulate_mineral(formula, standard_elements, n=5):
    if not isinstance(formula, list):
        formula = [formula]

    #elements = []
    #masses = []

    mineral_elements = {}
    def append(element, mass):
        if element in mineral_elements:
            mineral_elements[element] += mass
        else:
            mineral_elements[element] = mass

    for component in formula:
        if isinstance(component, str):
            for element, mass in get_formula(component, format="mass").items():
                append(element, np.ones(n)*mass)
        elif isinstance(component, dict):
            for molecule, percent in get_variable_percent(component['components'], n):
                for element, mass in get_formula(molecule, format="mass").items():
                    append(element, percent*mass)
        else:
            raise ValueError(f"{str(component)} is not a recognized format")

    # Calculate mass percents
    df = pd.DataFrame(mineral_elements)
    df.columns = [f"{element}_mass" for element in df.columns]
    df['mass'] = df.sum(axis=1)
    for element in mineral_elements:
        df[f"{element}_percent"] = df[f"{element}_mass"]/df['mass']

    # Convert to intensities
    for element in standard_elements:
        e = standard_elements[element]
        df[element] = (
            e['intercept'] + np.random.normal(scale=e['intercept'], size=n)
        )

        if f"{element}_percent" in df:
            df[element] += (
                e['coef']*df[f"{element}_percent"] +
                np.random.normal(scale=e['std'], size=n)
            )

        df[element] = np.clip(df[element], 0, None)


    '''mineral_elements = get_formula(component)
    for element, weight in mineral_elements.items():
        elements.append(element)
        masses.append(np.ones(n)*weight)'''

    #return elements, masses
    return df

def main(standards_dir, meteorite_dir, target_minerals_file, bits=32):

    characteristics = get_standards_characteristics(standards_dir, bits)
    target_minerals = load_target_minerals(target_minerals_file)
    print(characteristics)
    #

    for mineral, formula in target_minerals.items():
        print(mineral)

        print(simulate_mineral(formula, characteristics, 5).head())


def parse_args():
    """ Build argument parser and get parsed args """

    # Helper function to detect valid directories and files
    def valid_path(path_str):
        p = Path(path_str)
        if not p.exists():
            raise argparse.ArgumentTypeError(f"Could not find path {path_str}")
        return p

    def valid_dir(path_str):
        p = valid_path(path_str)
        if not p.is_dir():
            raise argparse.ArgumentTypeError(f"Path {path_str} is not a directory")
        return p

    def valid_file(path_str):
        p = valid_path(path_str)
        if p.is_dir():
            raise argparse.ArgumentTypeError(f"Path {path_str} is not a file")
        return p

    parser = argparse.ArgumentParser(description="Predict the mineral content of a "
                                                 "meteorite given spectrometer imagery.")
    parser.add_argument("standards_dir", type=valid_dir,
                        help="path to directory containing the standards")
    parser.add_argument("meteorite_dir", type=valid_dir,
                        help="path to directory containing the meteorite images")
    parser.add_argument("target_minerals_file", type=valid_file,
                        help="A YAML file containing the minerals to search for")
    parser.add_argument("--bits", type=int, choices=[8, 32], default=32,
                        help="image bit-depth to use (8 or 32)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
