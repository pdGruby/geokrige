# TransformerRasterio

## Description

```py
from geokrige.tools import TransformerRasterio
```

A class for handling operations on a [`rasterio`](https://rasterio.readthedocs.io/en/stable/) object (DatasetReader). 
This class makes it possible to **create effortlessly a mesh grid** for the `predict` method.

The class also allows to **save created mesh grids into a new GeoTIFF file** with the same attributes as loaded 
`rasterio` object. The interpolation results (mesh grids) can also be added into existing & loaded rasterio object.

## Methods

### `load`
**Load input data into the transformer.**

| Parameter |    Accepts    |                       Description                        |
|:---------:|:-------------:|:--------------------------------------------------------:|
|    `X`    | DatasetReader | An object on which the transformation will be performed. |

### `meshgrid`
**Create a mesh grid based on the loaded data.** Created mesh grid shape will correspond to loaded raster file.

| Parameter | Accepts | Description |
|:---------:|:-------:|:-----------:|
|    `-`    |    -    |      -      |

### `save`
**Save the provided layers to either a new raster file (GeoTIFF) or into the loaded DatasetReader object of the class 
instance.** The new raster file will have same attributes as the loaded DatasetReader object (crs, height, width etc.).

**Note that:** if the `inplace` parameter is False, then the `path` parameter must be specified.

| Parameter |    Accepts    |                                                                                    Description                                                                                     |
|:---------:|:-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `layers`  | List[ndarray] |                                                         List of numpy arrays representing layers (mesh grids) to be saved.                                                         |
|  `path`   |      str      |                                        A path to the raster file to be created, used only if the `inplace` parameter is `False`. Optional.                                         |
| `inplace` |     bool      | If `True`, the layers will be saved directly to the current DatasetReader object loaded into the class instance. Otherwise, a new raster file will be created. Default is `False`. |

## Attributes

### `X`

An object on which the transformation will be performed.

**Type:** DatasetReader

### `created_meshgrid`

A mesh grid which is a result of transformation.

**Type:** ndarray