# Utility Functions
import numpy as np
import seaborn as sns
from PIL import Image
import os

EPOCHS = 10
BATCH_SIZE = 10
HEIGHT = 256
WIDTH = 256
N_CLASSES = 13


def load_images(name, path):
    img = Image.open(os.path.join(path, name))
    img = np.array(img)

    image = img[:, :256]
    mask = img[:, 256:]

    return image, mask


def bin_image(mask):
    # Putting data into bins using NumPy, groups similar data points.
    bins = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
    new_mask = np.digitize(mask, bins)
    return new_mask


def get_segmentation_arr(image, classes, width=WIDTH, height=HEIGHT):
    seg_labels = np.zeros((height, width, classes))
    img = image[:, :, 0]

    for c in range(classes):
        seg_labels[:, :, c] = (img == c).astype(int)
    return seg_labels


def give_color_to_seg_img(seg, n_classes=N_CLASSES):
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    return seg_img
