#!/usr/bin/env python

import glob
import os
import sys
import time
from PIL import Image, ImageDraw
import argparse

from math import floor
import numpy as np
from scipy import stats
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Map Transition Generator")
    parser.add_argument("-m", "--map", type=str, help="Town name")
    parser.add_argument("-d", "--directory", type=str, default=None, help="Map directory (optional)")
    parser.add_argument("-r", "--resolution", type=int, help="Target resolution in pixels per meter")
    return parser.parse_args()


def downscale(array, scale_factor):
    # Define the size of the new downscaled array
    M, N = array.shape
    new_N = N // scale_factor
    new_M = M // scale_factor

    downscaled_array = np.zeros((new_M, new_N))

    # Iterate over the new downscaled array
    for i in range(new_N):
        for j in range(new_M):
            # Calculate the corresponding cells in the original array
            start_i = i * scale_factor
            start_j = j * scale_factor
            sub_array = array[start_i : start_i + scale_factor, start_j : start_j + scale_factor]

            # Find the most common value in these corresponding cells
            mode = stats.mode(sub_array, axis=None, keepdims=False).mode

            # Set the value of the cell in the downscaled array to this most common value
            downscaled_array[i, j] = mode

    return downscaled_array


material_to_identifier = {
    # (252, 233, 79)
    # (237, 212, 0)
    # (196, 160, 0)
    # (252, 175, 62)
    # (245, 121, 0)
    # (209, 92, 0)
    # (233, 185, 110)
    # (193, 125, 17)
    # (143, 89, 2)
    # (138, 226, 52)
    # (115, 210, 22)
    # (78, 154, 6)
    # (114, 159, 207)
    # (52, 101, 164)
    # (32, 74, 135)
    # (173, 127, 168)
    # (117, 80, 123)
    # (92, 53, 102)
    # (239, 41, 41)
    # (204, 0, 0)
    # (164, 0, 0)
    # (238, 238, 236)
    # (211, 215, 207)
    # (186, 189, 182)
    (136, 138, 133): "sidewalk",
    (85, 87, 83): "shoulder",
    (66, 62, 64): "parking",
    (46, 52, 54): "road",
    # Basic colors
    (255, 255, 255): "crosswalk",
    (0, 0, 0): "poles",
}

identifier_to_transition_probability = {
    "poles": 0,
    "other": 0.5,
    "parking": 0.05,
    "shoulder": 0.05,
    "road": 0.05,
    "crosswalk": 1.0,
    "sidewalk": 1.0,
}


def decolorize(array):
    # Create a new array with the same shape as the input array
    new_array = np.zeros(array.shape[:2])

    # Iterate over the input array
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # Set the value of the cell in the new array to the index of the material identifier
            try:
                new_array[i, j] = identifier_to_transition_probability[material_to_identifier[tuple(array[i, j])]]
            except KeyError:
                new_array[i, j] = identifier_to_transition_probability["other"]

    return new_array


def main():
    #
    args = parse_args()
    input_name = args.file
    target_resolution = args.resolution
    output_name = sys.os.path.splitext(args.input)[0]

    # parse the image origin and resolution from the image name, where
    # image name format: town_name-origin_x-origin_y-resolution.png
    image_name = os.path.splitext(os.path.basename(input_name))[0].split("_")

    # get the origin and resolution
    origin_x = float(image_name[1])
    origin_y = float(image_name[2])
    resolution = float(image_name[3])

    step_size = int(resolution / target_resolution)

    # have a map with dimensions dim_pixel_x, dim_pixel_y and an offset of origin_x, origin_y
    # then any location in world coordinates (x, y) corresponds to the pixel location
    # (x - origin_x) * resolution, (y - origin_y) * resolution  and in reverse
    # (pixel_x / resolution + origin_x, pixel_y / resolution + origin_y)
    def world_to_pixel(x, y, origin, resolution):
        return (int((x - origin[0]) * resolution), int((y - origin[1]) * resolution))

    def pixel_to_world(pixel_x, pixel_y, origin, resolution):
        return (pixel_x / resolution + origin[0], pixel_y / resolution + origin[1])

    # Load the image
    img = Image.open(input_name).convert("RGB")
    img_data = np.array(img)

    material_grid = decolorize(img_data)
    downscaled = downscale(material_grid, step_size)

    # with the new image, we now need to translate it into a probability grid, where each cell has a probability
    # of transitioning to a neighboring cell. We will use a 3x3 kernel to calculate the probabilities
    # for each cell.  The kernel is normalized to sum to 1.0.

    # TODO: include probabilities based on oriEntation and perhaps some sort of lattice structure

    # for each cell in the downscaled image, generate a 9x1 vector of probabilities.
    transition_probabilities = np.zeros((downscaled.shape[0], downscaled.shape[1], 9))

    # pad the downscaled image with zeros on the edges
    downscaled = np.pad(downscaled, 1)

    # TODO: we are hard coding a couple things here:  cells are 0.5m, and the pedestrian moves at 0.15m
    #       per 0.1s time interval.  This means there is a fairly high probability the pedestrian will remain
    #       in the same grid cell in any given time interval.
    PEDESTRIAN_REMAIN_PROBABILITY = 0.4

    with np.errstate(divide="raise", invalid="raise"):
        M, N = downscaled.shape
        for m in range(1, M - 1):
            for n in range(1, N - 1):
                # get the 3x3 grid around the cell
                grid = np.array(downscaled[m - 1 : m + 2, n - 1 : n + 2])
                total = np.sum(grid) - grid[1, 1]
                grid[1, 1] = (
                    PEDESTRIAN_REMAIN_PROBABILITY / (1 - PEDESTRIAN_REMAIN_PROBABILITY)
                ) * total  # probability of staying in the same cell
                total += grid[1, 1]

                # normalize the grid
                if total > 0:
                    grid = grid / total

                transition_probabilities[n - 1, m - 1, :] = grid.flatten()

    # save the transition probabilities
    np.save(f"{output_name}-transitions.npy", transition_probabilities)

    # write the statistics to a json file
    stats = {
        "origin": [origin_x, origin_y],
        "resolution": target_resolution,
        "_comment_resolution": "Resolution is in pixels per meter",
    }
    with open(f"{output_name}-stats.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    main()
