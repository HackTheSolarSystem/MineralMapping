import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from lib import get_standards_characteristics, load_target_minerals, get_formula, load_images

def get_variable_percent(formula, n, epsilon=.000001):
    """
    When a mineral has components that can have varying amounts of elements,
    simulate n examples of the different percentages. The results should add
    up to 1.

    For example, Olivine has a component:

        {Fe: [0, 1], Mg: [0, 1]}

    This means that it can be all Fe, all Mg, or somewhere inbetween.

    Returns a list of tuples with the element and a numpy array of percenages.

    For example:

    [
        ('Fe', np.array([1, .5, .3, .7, 0, ...])),
        ('Mg', np.array([0, .5, .7, .3, 1, ...]))
    ]
    """
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

def simulate_mass(formula, n):
    """
    Given a mineral formula, return n simulated examples.

    The formula can either be a string as a chemical formula, or a list of
    strings and dicts. See `target_minerals.yaml` for examples. In the case that
    the formula is just a chemical formula string, all n examples will be the
    same.

    Returns a DataFrame where each row is an example and there are two columns
    for each element in the mineral: element_mass which contains the mass and
    element_percent which has that elements perentage of the whole mass of that
    row.
    """
    if not isinstance(formula, list):
        formula = [formula]

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
            if 'quantity' in component:
                quantity = component['quantity']
                if isinstance(quantity, list):
                    quantity = np.random.randint(quantity[0], quantity[1]+1)
            else:
                quantity = 1

            for molecule, percent in get_variable_percent(component['components'], n):
                for element, mass in get_formula(molecule, format="mass").items():
                    append(element, percent*mass*quantity)
        else:
            raise ValueError(f"{str(component)} is not a recognized format")

    # Calculate mass percents
    df = pd.DataFrame(mineral_elements)
    df.columns = [f"{element}_mass" for element in df.columns]
    df['mass'] = df.sum(axis=1)
    for element in mineral_elements:
        df[f"{element}_percent"] = df[f"{element}_mass"]/df['mass']

    return df

def simulate_mineral(mineral, formula, elements, n=100, noise=10):
    """
    Simulate a mineral's intensities as if it were scanned by the electron
    microprobe. Return a DataFrame where each row is one simulated example
    of that mineral.

    Parameters
    ----------
    mineral: str
        The name of the mineral
    formula: str or list or dict
        The formula for the mineral using the format in `target_minerals.yaml`
    elements: dict
        A dict describing the characteristics of each element in the electron
        microprobe scan. Obtained from lib.get_standards_characteristics
    n: int
        The number of examples to create. (Default 5)
    noise: number
        The amount of noise to add to each element channel. More noise will
        allow the classifier to have more tolerance when classifying minerals
        which contain trace amounts of unexpected elements. (Default 10)
    """
    df = simulate_mass(formula, n)

    # Convert to intensities
    for element in elements:
        e = elements[element]
        df[element] = (
            e['intercept'] + np.clip(
                np.random.normal(scale=e['noise']*noise, size=n),
                0, None
            )
        )

        if f"{element}_percent" in df:
            df[element] += (
                e['coef']*df[f"{element}_percent"] +
                np.random.normal(scale=e['std'], size=n)
            )

        df[element] = np.clip(df[element], 0, None)

    df['mineral'] = mineral
    return df

def main(standards_dir, meteorite_dir, target_minerals_file, output_dir,
         title=None, bits=32, mask=None, n=100, unknown_n=None, noise=10,
         model=None):
    characteristics = get_standards_characteristics(standards_dir, bits)
    target_minerals = load_target_minerals(target_minerals_file)
    #print(characteristics)
    elements = list(characteristics.keys())

    mineral_dfs = []
    for mineral, formula in target_minerals.items():
        #print(mineral)

        df = simulate_mineral(mineral, formula, characteristics, n)
        #print(df.head())
        mineral_dfs.append(df[elements + ['mineral']])

    df = pd.concat(mineral_dfs)

    if unknown_n is None:
        unknown_n = n
    if unknown_n > 0:
        unknown = pd.DataFrame(np.clip(
            np.hstack([
                np.random.uniform(-m, m, (unknown_n, 1)) +
                np.random.normal(scale=noise, size=(unknown_n, 1))
                for m in df[elements].max(axis=0)
            ]), 0, None), columns=elements
        )
        unknown['mineral'] = 'Unknown'
        df = pd.concat([df, unknown])


    X = df[elements].values
    Y = df['mineral']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

    print("Training Classifier...")
    if model is None:
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()

    model.fit(X_train, Y_train)

    print("Training Accuracy:", (model.predict(X_train) == Y_train).mean())
    print("Testing Accuracy:", (model.predict(X_test) == Y_test).mean())

    meteorite_df, meteorite_shape = load_images(meteorite_dir, bits, mask)
    x = meteorite_df[elements].values
    meteorite_df['mineral'] = model.predict(x)
    #print(meteorite_df.head(20))

    if mask:
        minerals = sorted(meteorite_df[meteorite_df['mask'] > 0]['mineral'].unique())
    else:
        minerals = sorted(meteorite_df['mineral'].unique())

    results = meteorite_df.merge(
        pd.Series(
            minerals, name='mineral'
        ).reset_index().rename(columns={'index': 'mineral_index'}),
        on='mineral'
    ).sort_values('order')

    figure, ax = plt.subplots(figsize=(20,20))
    norm = plt.Normalize(0, len(minerals)-1)
    cmap = plt.cm.get_cmap('jet')
    rgb = cmap(norm(results['mineral_index'].values.reshape(meteorite_shape)))
    if mask:
        rgb[..., -1] = results['mask'].values.reshape(meteorite_shape)
    im = ax.imshow(rgb)

    colors = [cmap(norm(i)) for i in range(len(minerals))]
    patches = [
        mpatches.Patch(
            color=colors[i], label=minerals[i]
        ) for i in range(len(minerals))
    ]
    ax.legend(
        handles=patches, bbox_to_anchor=(1.3, .5, 0, 0),
        loc=5, borderaxespad=0., fontsize=30
    )

    if title:
        figure.suptitle(title, fontsize=30, y=.91)

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    plt.savefig(
        output_dir / 'figure.png',
        facecolor='white', transparent=True, frameon=False, bbox_inches='tight'
    )

    results.groupby('mineral').count()['mineral_index'].sort_values(
        ascending=False
    ).to_csv(output_dir / 'mineral_counts.csv')

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

    def valid_model(model):
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC
        from sklearn.ensemble import (
            RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
        )
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier

        if (model is None) or (model == "GaussianNB"):
            return GaussianNB()
        elif model == "RandomForest":
            return RandomForestClassifier(50, max_depth=10)
        else:
            return eval(model)

    parser = argparse.ArgumentParser(description="Predict the mineral content of a "
                                                 "meteorite given spectrometer imagery.")
    parser.add_argument("standards_dir", type=valid_dir,
                        help="path to directory containing the standards")
    parser.add_argument("meteorite_dir", type=valid_dir,
                        help="path to directory containing the meteorite images")
    parser.add_argument("target_minerals_file", type=valid_file,
                        help="A YAML file containing the minerals to search for")
    parser.add_argument("output_dir", type=str,
                        help="The directory to write the outputs to.")
    parser.add_argument("--mask", type=valid_file, default=None,
                        help="An optional mask to use for the meteorite.")
    parser.add_argument("--title", type=str, default=None,
                        help="An optional title to put on the output image.")

    parser.add_argument("--n", type=int, default=100,
                        help="""The number of samples to simulate. (Default 100)
                                The higher the number, the more robust the
                                predictions, but the longer it will take.""")
    parser.add_argument("--unknown_n", type=int, default=None,
                        help="""The number of samples to use for "Unknown."
                                The higher the number relative to n, the more
                                likely that a pixel will be classified as
                                "Unknown". Set to 0 to disable Unknown
                                classifications. (Default to the same as n.)
                        """)
    parser.add_argument("--model", type=valid_model, default=None,
                        help="""A classification algorithm to use. Either
                                "RandomForest" or "GaussianNB" or a string
                                which can be evaluated to a sklearn model
                                such as "KNeighborsClassifier(10)".
                                (Default GaussianNB)""")

    parser.add_argument("--bits", type=int, choices=[8, 32], default=32,
                        help="image bit-depth to use (8 or 32)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
