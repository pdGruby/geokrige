from typing import List

import rasterio
from rasterio.io import DatasetReader
import numpy as np


class TransformerRasterio:
    """
    A class for handling operations on a Rasterio object (DatasetReader).

    Attributes
    ----------
    X : DatasetReader
        An object on which the transformation will be performed.
    created_meshgrid : np.ndarray
        A mesh grid which is a result of transformation.

    Methods
    -------
    load(X)
        Load input data into the transformer.
    meshgrid(density)
        Create a mesh grid based on the loaded data.
    save(layers, path)
        Save the provided layers to a new raster file.
    """
    def __init__(self):
        self.X = None
        self.created_meshgrid = None

    def load(self, X: DatasetReader) -> None:
        """
        Load input data into the transformer.

        Parameters
        ----------
        X : DatasetReader
            An object on which the transformation will be performed.

        Returns
        -------
        None
        """
        if not isinstance(X, DatasetReader):
            raise TypeError(f"Invalid input for the 'X' parameter! Should be DatasetReader got {X} instead")
        self.X = X

    def meshgrid(self) -> List[np.ndarray]:
        """
        Create a mesh grid based on the loaded data.

        Returns
        -------
        List[np.ndarray]
            Generated meshgrid.
        """
        transform = self.X.transform
        width, height = self.X.width, self.X.height
        x = np.linspace(transform[2], transform[2] + transform[0] * (width - 1), width)
        y = np.linspace(transform[5], transform[5] + transform[4] * (height - 1), height)

        meshgrid = np.meshgrid(x, y)
        self.created_meshgrid = meshgrid

        return meshgrid

    def save(self, layers: List[np.ndarray], path: str) -> None:
        """
        Save the provided layers to a new raster file (GeoTIFF).
        The new raster file will have same attributes as the loaded
        DatasetReader object (crs, height, width etc.).

        Parameters
        ----------
        layers : List[np.ndarray]
            List of numpy arrays representing layers (mesh grids) to be
            saved.
        path : str
            A path to the raster file that will be created.

        Returns
        -------
        None
        """
        transform = self.X.transform
        width, height = self.X.width, self.X.height

        dst = rasterio.open(path, 'w', driver='GTiff', height=height, width=width, count=len(layers),
                            dtype='float32', crs=self.X.crs, transform=transform)
        i = 0
        for layer in layers:
            i += 1
            dst.write(layer, i)

        dst.close()

        print(f"Successfully saved {i} new layers into the {path} file.")
