from skimage import transform
from skimage import exposure
from skimage.util import random_noise
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import random

################# IMAGE TRANSFORMATION FUNCTIONs ####################.

def random_rotate(image):
    random_degree = random.uniform(-10, 10)
    new_img = 255 * transform.rotate(image, random_degree)
    new_img = new_img.astype(np.uint8)
    return new_img


def flip_horizontal(image, prob=1.):
    random_prob = random.uniform(0., 1.)
    if random_prob < prob:
        new_img = np.fliplr(image)
    return new_img


def random_shear(image):
    random_degree = random.uniform(-0.1, 0.1)
    afine_tf = transform.AffineTransform(shear=random_degree)
    new_img = 255 * transform.warp(image, inverse_map=afine_tf)
    new_img = new_img.astype(np.uint8)
    return new_img


def change_contrast(image, percent_change=(0, 15)):
    percent_change = random.uniform(percent_change[0], percent_change[1])
    v_min, v_max = np.percentile(image, (0. + percent_change, 100. - percent_change))
    new_img = exposure.rescale_intensity(image, in_range=(v_min, v_max))
    new_img = new_img.astype(np.uint8)
    return new_img


def gamma_correction(image, gamma_range=(0.7, 1.0)):
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    new_img = exposure.adjust_gamma(image, gamma=gamma, gain=random.uniform(0.8, 1.0))
    new_img = new_img.astype(np.uint8)
    return new_img


def blur_image(image):
    new_img = ndimage.uniform_filter(image, size=(5, 5, 1))
    new_img = new_img.astype(np.uint8)
    return new_img


def add_noise(image):
    new_img = 255 * random_noise(image)
    new_img = new_img.astype(np.uint8)
    return new_img

def no_action(image):
    return image

transforms = {      0: random_rotate,
                    1: flip_horizontal,
                    2: random_shear,
                    3: change_contrast,
                    4: gamma_correction,
                    5: blur_image,
                    6: add_noise,
                    7: no_action
             }





