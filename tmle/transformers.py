import numpy as np
import skimage

from typing import Optional
from skimage.feature import hog
from sklearn.base import TransformerMixin


class HOGTransformer(TransformerMixin):

    def __init__(
            self,
            img_shape: tuple = (224, 224),
            orientations: int = 9,
            pixels_per_cell: tuple = (8, 8),
            cells_per_block: tuple = (3, 3),
            block_norm: Optional[bool] = None,
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
        """Fit method."""
        return self

    def transform(self, X: np.ndarray):
        """Transform images by applying HOG."""
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
                visualise=self.visualize,
                transform_sqrt=self.transform_sqrt,
                feature_vector=self.feature_vector,
                multichannel=self.multichannel
            )
        return X_transformed
