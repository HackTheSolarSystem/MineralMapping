#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import numpy
from PIL import Image


def get_masked_array(src_array, mask_array):
    maxed_mask_array = mask_array.copy()
    for i in range(len(mask_array)):
        for j in range(len(mask_array[0])):
            maxed_mask_array[i][j] = 1 if mask_array[i][j] > 0 else 0

    masked_array = numpy.multiply(src_array, maxed_mask_array)
    return masked_array


def main(source_path, mask_path, out_path):
    src_img = Image.open(source_path)
    mask_img = Image.open(mask_path)

    src_array = numpy.array(src_img)
    mask_array = numpy.array(mask_img)

    masked_array = get_masked_array(src_array, mask_array)
    masked_img = Image.fromarray(masked_array)
    masked_img.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", dest="source_path", required=True)
    parser.add_argument("--mask", dest="mask_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
