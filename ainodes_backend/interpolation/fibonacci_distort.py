import random

import cv2
import numpy as np


def fibonacci_distortion(image, seed=0):
    random.seed(seed)
    cols, rows = image.shape[:2]
    x_map = np.arange(rows).reshape(-1, 1).repeat(cols, axis=1).astype(np.float32)
    y_map = np.arange(cols).reshape(1, -1).repeat(rows, axis=0).astype(np.float32)

    x_center, y_center = int(rows / 2), int(cols / 2)
    max_distance = np.sqrt(x_center ** 2 + y_center ** 2)

    x_map = x_map - x_center
    y_map = y_map - y_center

    distance = np.sqrt(x_map ** 2 + y_map ** 2)
    angle = np.arctan2(y_map, x_map)

    num_mirrors = random.randint(8, 16)
    mirrors = np.linspace(0, 2 * np.pi, num_mirrors + 1)[:-1]

    x_map_list = [x_map * np.cos(mirror + distance * 0.03) + y_map * np.sin(mirror + distance * 0.03) for mirror in
                  mirrors]
    y_map_list = [-x_map * np.sin(mirror + distance * 0.03) + y_map * np.cos(mirror + distance * 0.03) for mirror in
                  mirrors]

    x_map = np.concatenate(x_map_list, axis=1)
    y_map = np.concatenate(y_map_list, axis=1)

    x_map = x_map + x_center
    y_map = y_map + y_center

    distorted_image = cv2.remap(image, y_map, x_map, cv2.INTER_LINEAR)
    return distorted_image
def fibonacci_spiral_distortion_3(image, strength=0.1):
    strength = 2500


    cols, rows = image.shape[:2]
    x_map = np.arange(rows).reshape(-1, 1).repeat(cols, axis=1).astype(np.float32)
    y_map = np.arange(cols).reshape(1, -1).repeat(rows, axis=0).astype(np.float32)

    x_center, y_center = int(rows / 2), int(cols / 2)
    max_distance = np.sqrt(x_center ** 2 + y_center ** 2)

    x_map = x_map - x_center
    y_map = y_map - y_center

    distance = np.sqrt(x_map ** 2 + y_map ** 2)
    angle = np.arctan2(y_map, x_map)

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = distance * golden_angle / max_distance

    x_map = x_map + strength * np.cos(theta) * np.cos(angle)
    y_map = y_map + strength * np.cos(theta) * np.sin(angle)

    x_map = x_map + x_center
    y_map = y_map + y_center

    distorted_image = cv2.remap(image, y_map, x_map, cv2.INTER_LINEAR)
    return distorted_image


def fibonacci_spiral_distortion_2(image, strength=0.1):
    rows, cols = image.shape[:2]
    x_map = np.arange(rows).reshape(-1, 1).repeat(cols, axis=1).astype(np.float32)
    y_map = np.arange(cols).reshape(1, -1).repeat(rows, axis=0).astype(np.float32)

    x_center, y_center = int(rows/2), int(cols/2)
    max_distance = np.sqrt(x_center**2 + y_center**2)

    x_map = x_map - x_center
    y_map = y_map - y_center

    distance = np.sqrt(x_map**2 + y_map**2)
    angle = np.arctan2(y_map, x_map)

    x_map = x_map + strength * max_distance * np.sin(angle + distance * 0.1)
    y_map = y_map + strength * max_distance * np.cos(angle + distance * 0.1)

    x_map = x_map + x_center
    y_map = y_map + y_center

    distorted_image = cv2.remap(image, x_map, y_map, cv2.INTER_LINEAR)
    return distorted_image

def fibonacci_spiral_distortion_slow(image, strength=0.1):
    rows, cols = image.shape[:2]
    x_map = np.zeros((rows, cols), np.float32)
    y_map = np.zeros((rows, cols), np.float32)

    x_center, y_center = int(rows/2), int(cols/2)
    max_distance = np.sqrt(x_center**2 + y_center**2)

    for i in range(rows):
        for j in range(cols):
            x_map[i, j] = i
            y_map[i, j] = j
            distance = np.sqrt((i-x_center)**2 + (j-y_center)**2)
            angle = np.arctan2(j-y_center, i-x_center)
            x_map[i, j] += strength * max_distance * np.sin(angle + distance * 0.1)
            y_map[i, j] += strength * max_distance * np.cos(angle + distance * 0.1)

    distorted_image = cv2.remap(image, x_map, y_map, cv2.INTER_LINEAR)
    return distorted_image


def interesting_distortion(image, strength=0.1):
    cols, rows = image.shape[:2]
    x_map = np.arange(rows).reshape(-1, 1).repeat(cols, axis=1).astype(np.float32)
    y_map = np.arange(cols).reshape(1, -1).repeat(rows, axis=0).astype(np.float32)

    x_center, y_center = int(rows/2), int(cols/2)
    max_distance = np.sqrt(x_center**2 + y_center**2)

    x_map = x_map - x_center
    y_map = y_map - y_center

    distance = np.sqrt(x_map**2 + y_map**2)
    angle = np.arctan2(y_map, x_map)

    x_map = x_map + strength * max_distance * np.sin(angle + distance * 0.1)
    y_map = y_map + strength * max_distance * np.cos(angle + distance * 0.1)

    x_map = x_map + x_center
    y_map = y_map + y_center

    x_map = np.repeat(x_map[..., np.newaxis], 3, axis=-1)
    y_map = np.repeat(y_map[..., np.newaxis], 3, axis=-1)

    distorted_image = cv2.remap(image.T, y_map.T, x_map.T, cv2.INTER_LINEAR)
    return distorted_image.T