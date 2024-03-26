# TransformerRasterio

## Description

```py
from geokrige.tools import TransformerRasterio
```

A class for handling operations on a [`rasterio`](https://rasterio.readthedocs.io/en/stable/) object (DatasetReader). 
This class makes it possible to **create effortlessly a mesh grid** for the `predict` method.

The class also allows to **save created mesh grids into a new GeoTIFF file** with the same attributes as loaded 
`rasterio` object.

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
**Save the provided layers to a new raster file (GeoTIFF).** The new raster file will have same attributes as the loaded 
DatasetReader object (crs, height, width etc.).

| Parameter |    Accepts    |                            Description                             |
|:---------:|:-------------:|:------------------------------------------------------------------:|
| `layers`  | List[ndarray] | List of numpy arrays representing layers (mesh grids) to be saved. |
|  `path`   |      str      |          A path to the raster file that will be created.           |

## Attributes

### `X`

An object on which the transformation will be performed.

**Type:** DatasetReader

### `created_meshgrid`

A mesh grid which is a result of transformation.

**Type:** ndarray