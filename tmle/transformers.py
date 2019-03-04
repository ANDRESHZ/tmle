import numpy as np
import torch
import torch.nn as nn
import torchvision

from typing import Optional, Tuple
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from .dataloaders import ImageFoldersDataset


class CNNFeatures(object):

    def __init__(self, n_features: int, rm_top_layers: int):

        self.n_features = n_features
        self.rm_top_layers = rm_top_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_features(
            self,
            model: torchvision.models,
            dataset: ImageFoldersDataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param model:
        :param dataset:
        :return:
        """
        model_to_extractor = self._remove_n_top_layers(model)
        model_to_extractor.eval()
        model_to_extractor = model_to_extractor.to(self.device)
        # define placeholder for features and labels
        X_codes, y_codes = np.zeros((len(dataset), self.n_features)), np.zeros(len(dataset))
        for data_idx, data in enumerate(dataset.dataset):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = model_to_extractor(inputs).view(-1, self.n_features)
            X_codes[data_idx] = outputs.cpu().numpy()
            y_codes[data_idx] = labels.cpu().numpy()
        return X_codes, y_codes

    def _remove_n_top_layers(
            self,
            model: torchvision.models
    ) -> torch.nn.modules.container.Sequential:
        """

        :param model:
        :return:
        """
        modules = list(model.children())[:-self.rm_top_layers]
        return nn.Sequential(*modules).to(self.device)


class HOGTransformer(BaseEstimator, TransformerMixin):
    """
    `HOGTransformer` implements method that allow *HOG Descriptor* to be included
    in the `sklearn.Pipeline`. Thanks to this *HOG Descriptor* can be easily integraded
    with other `Transformers` and `Classifiers` from `sklearn`.

    :param img_shape: width and height of images (number of channels is not required).
    :param orientations: number of bins in histograms.
    :param pixel_per_cell: number of pixels in given cell.
    :param cells_per_block: number of cells present in one block.
    :param visualize:
    :param transform_sqrt:
    :param feature_vector:
    :param multichannel:
    """
    def __init__(
            self,
            img_shape: tuple = (224, 224),
            orientations: int = 9,
            pixels_per_cell: tuple = (8, 8),
            cells_per_block: tuple = (3, 3),
            block_norm: Optional[bool] = 'L2-Hys',
            visualize: bool = False,
            transform_sqrt: bool = False,
            feature_vector: bool = True,
            multichannel: Optional[bool] = None
    ):

        self.img_shape = img_shape
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.visualize = visualize
        self.transform_sqrt = transform_sqrt
        self.feature_vector = feature_vector
        self.multichannel = multichannel

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit method was implemented only in order to achieve consistency with `sklearn` templates."""
        return self

    def transform(self, X: np.ndarray):
        """ Transform will return the histograms of oriented gradients for images passed
        as `X`. Output can be further passed to other `sklearn`'s classes.

        :param X: array with images, ie. (batch_size, 224, 224, 3).
        :return: array with shape (batch_size, n_blocks_row * n_blocks_col * cells_per_block_row *
            cells_per_block_col * orientations).
        """
        n_blocks_row = int(self.img_shape[0] / self.pixels_per_cell[0] - (self.cells_per_block[0] - 1))
        n_blocks_col = int(self.img_shape[1] / self.pixels_per_cell[1] - (self.cells_per_block[1] - 1))
        X_transformed = np.zeros((
            len(X),
            n_blocks_row * n_blocks_col * self.cells_per_block[0] * self.cells_per_block[1] * self.orientations
        ))
        for x_idx, x in enumerate(X):
            X_transformed[x_idx, :] = hog(
                x,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=self.visualize,
                transform_sqrt=self.transform_sqrt,
                feature_vector=self.feature_vector,
                multichannel=self.multichannel
            )
        return X_transformed