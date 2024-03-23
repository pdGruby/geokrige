from typing import List

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from shapely.geometry import Point, Polygon, MultiPolygon


class TransformerGDF:
    """
    A class for handling operations on a GeoDataFrame object.

    Attributes
    ----------
    X : GeoDataFrame
        An object on which the transformation will be performed.
    created_meshgrid : np.ndarray
        A mesh grid which is a result of transformation.

    Methods
    -------
    load(X)
        Load input data into the transformer.
    meshgrid(density)
        Create a meshgrid based on the loaded data.
    mask(meshgrid, gdf_object)
        Create a mask for the meshgrid based on loaded GeoDataFrame object.
    """
    def __init__(self):
        self.X = None
        self.created_meshgrid = None

    def load(self, X: GeoDataFrame) -> None:
        """
        Load input data into the transformer.

        Parameters
        ----------
        X : GeoDataFrame
            An object on which the transformation will be performed.

        Returns
        -------
        None
        """
        if not isinstance(X, GeoDataFrame):
            raise TypeError(f"Invalid input for the 'X' parameter! Should be GeoDataFrame got {X} instead")
        self.X = X

    def meshgrid(self, density: float = 1.0) -> List[np.ndarray]:
        """
        Create a mesh grid based on the loaded data.

        Parameters
        ----------
        density : float
            Grid density for 2D space. It can be increased if the returned
            mesh grid is too sparse, and decreased if it is too dense. Default
            is 1.0.

        Returns
        -------
        List[np.ndarray]
            Generated meshgrid.
        """
        if density <= 0:
            raise ValueError("The 'grid_density' value must be a positive float!")

        are_all_polygons = self.X['geometry'].apply(lambda geom: isinstance(geom, (Polygon, MultiPolygon))).all()
        if not are_all_polygons:
            raise ValueError('Not all geometry objects in the GeoDataFrame are of POLYGON type.')

        min_x, min_y, max_x, max_y = self.X.total_bounds
        x_range = max_x - min_x
        y_range = max_y - min_y

        if x_range > y_range:
            x_density_coef = density
            y_density_coef = density * (x_range / y_range)
        else:
            x_density_coef = density * (y_range / x_range)
            y_density_coef = density

        x_dots = int(round(100 * x_density_coef))
        y_dots = int(round(100 * y_density_coef))
        lon = np.linspace(min_x, max_x, num=x_dots)
        lat = np.linspace(min_y, max_y, num=y_dots)

        meshgrid = np.meshgrid(lon, lat)
        self.created_meshgrid = meshgrid
        return meshgrid

    def mask(self, meshgrid: List[np.ndarray] = None) -> np.ndarray:
        """
        Create a mask for the mesh grid based on loaded GeoDataFrame object.
        If the 'meshgrid' argument is not provided, a class attribute will
        be used (created by the 'meshgrid' method)

        Parameters
        ----------
        meshgrid : List[np.ndarray], optional
            Meshgrid to be masked. If None, uses internally stored meshgrid (created
            in the 'meshgrid' method).

        Returns
        -------
        np.ndarray
            Mask for the meshgrid. The points of the mesh grid that are within
            the GeoDataFrame polygons will have True value. The points that are
            not, will have False value.
        """
        if not meshgrid:
            meshgrid = self.created_meshgrid

        if meshgrid is None:
            raise ValueError("The 'meshgrid' parameter is None & the class 'created_meshgrid' attribute is also None. "
                             "If the 'GDFTransformer' is intended to generate a mesh grid, call the 'meshgrid' method "
                             "first. If the 'GDFTransformer' is not intended to generate a mesh grid, provide it to "
                             "the 'meshgrid' parameter.")

        geometry = [Point(lon, lat) for lon, lat in zip(*[coord.ravel() for coord in meshgrid])]
        meshgrid_as_gdf = GeoDataFrame({'geometry': geometry}, crs=str(self.X.crs))

        clipped_data = meshgrid_as_gdf.clip(self.X).copy()
        mask = np.isin(meshgrid_as_gdf.index.to_numpy(), clipped_data.index)

        return mask.reshape(meshgrid[0].shape)
