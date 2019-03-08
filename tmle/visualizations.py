import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from PIL import Image
from torchvision import transforms


LABELS = [
    'bird',
    'boat',
    'cat',
    'dog',
    'flower',
    'frog',
    'jumbojet',
    'mushroom',
    'sportscar',
    'tree'
]


def plot_grid_of_images(
        path_to_images_folder: str,
        transform: torchvision.transforms,
        save: bool = False,
        save_path: str = 'images_grid.png',
        figsize: tuple = (36, 36),
        nrows: int = 10,
        ncols: int = 11,
        fontsize: int = 32
) -> None:
    """Plot grid of images from ImagesFolder.

    :param path_to_images_folder: path to folder with images.
    :param transform: transformations which will be applied to images.
    :param figsize: size of a figure.
    :param nrows: number of rows (corresponding to number of labels).
    :param ncols: number of columns (corresponding to number of images for given label).
    :param fontsize: size of a font which annotates labels.
    :return: plot of images.
    """
    f, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    for l, label in enumerate(glob.glob(os.path.join(path_to_images_folder, '*'))):
        images = glob.glob(os.path.join(label, '*'))
        images_sample = np.array(images)[np.random.randint(0, len(images), ncols - 1)]
        # annotate labels
        ax[l][0].text(0, 0.5, os.path.basename(label), fontsize=fontsize)
        ax[l][0].axis('off')
        # plot images
        for i, image in enumerate(images_sample):
            image = np.asarray(transform(Image.open(image)))
            ax[l][i+1].imshow(image)
            ax[l][i+1].axis('off')
    if save:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(
        matrix: np.ndarray,
        labels: list = LABELS,
        figsize: tuple = (10, 10)
) -> None:
    """Plot confusion matrix.

    :param matrix: confusion matrix (sklearn.metrics.confusion_matrix).
    :param labels: labels.
    :param figsize: size of a figure.
    :return: plot of confusion matrix.
    """
    f, ax = plt.subplots(figsize=figsize)
    ax.imshow(matrix)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label'
    )
    formatter = '.2f'
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, format(matrix[row, col], formatter),
                    ha='center', va='center', color='white')
    f.tight_layout()
    plt.show()