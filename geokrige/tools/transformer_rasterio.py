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
    save(layers, inplace, path)
        Save the provided layers to either a new raster file
        (GeoTIFF) or into the loaded raster.
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

    def save(self, layers: List[np.ndarray], path: str = None, inplace: bool = False) -> None:
        """
        Save the provided layers to either a new raster file (GeoTIFF)
        or into the loaded DatasetReader object of the class instance.
        The new raster file will have same attributes as the loaded
        DatasetReader object (crs, height, width etc.).

        Note that
        ---------
        If the 'inplace' parameter is False, then the 'path' parameter
        must be specified.

        Parameters
        ----------
        layers : List[np.ndarray]
            List of numpy arrays representing layers (mesh grids) to be
            saved.
        path : str, optional
            A path to the raster file to be created, used only if the 'inplace'
            parameter is False.
        inplace : bool, optional
            If True, the layers will be saved directly to the current
            DatasetReader object loaded into the class instance. Otherwise,
            a new raster file will be created. Default is False.

        Raises
        ------
        ValueError
            If the combination of arguments is invalid.

        Returns
        -------
        None
        """
        transform = self.X.transform
        width, height = self.X.width, self.X.height

        if not inplace and path:
            dst = rasterio.open(path, 'w', driver='GTiff', height=height, width=width, count=len(layers),
                                dtype='float32', crs=self.X.crs, transform=transform)
        elif inplace:
            dst = self.X
        else:
            raise ValueError("Invalid combination of the arguments. If the 'inplace' parameter is set to False, then "
                             f"the 'path' argument must be specified. Currently the 'path' value is: {path}.")

        i = 0
        for layer in layers:
            i += 1
            dst.write(layer, i)

        if not inplace:
            dst.close()

        print(f"Successfully saved {i} new layers into the {path if not inplace else 'rasterio object.'}")
