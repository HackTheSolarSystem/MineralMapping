import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Predict the mineral content of a meteorite given spectrometer imagary.')
parser.add_argument('standards', type=str, help='a directory containing the standards.')
parser.add_argument('--bits', default='32', help='image bit-depth to use (8 or 32)')

args = parser.parse_args()
print(args)

standard_path = Path(args.standards)

standard_masks = [
    i for i in list(standard_path.glob('*_mask.tif')) if 'obj' not in i.name
]

print(standard_masks)
