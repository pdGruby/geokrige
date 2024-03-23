# TransformerGDF

## Description

```py
from geokrige.tools import TransformerGDF
```

A class for handling operations on a [`GeoDataFrame`](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html) 
object. This class makes it possible to **create effortlessly a mesh grid** for the `predict` method. 

Additionally, a user can also **create a mask for generated mesh grid or other passed mesh grids** on the basis of 
POLYGONS incorporated in loaded `GeoDataFrame` object – this can be used in order to clip interpolated data so that it 
covers only the regions that are within the POLYGONS boundaries.

## Methods

### `load`
**Load input data into the transformer.**

| Parameter |   Accepts    |                       Description                        |
|:---------:|:------------:|:--------------------------------------------------------:|
|    `X`    | GeoDataFrame | An object on which the transformation will be performed. |

### `meshgrid`
**Create a mesh grid based on the loaded data.**

| Parameter | Accepts |                                                                 Description                                                                 |
|:---------:|:-------:|:-------------------------------------------------------------------------------------------------------------------------------------------:|
| `density` |  float  | Grid density for 2D space. It can be increased if the returned mesh grid is too sparse, and decreased if it is too dense. Default is `1.0`. |

### `mask`
**Create a mask for the mesh grid based on loaded GeoDataFrame object.** If the `meshgrid` argument is not provided, a 
class attribute will be used (created by the `meshgrid` method)

| Parameter  | Accepts |                                              Description                                              |
|:----------:|:-------:|:-----------------------------------------------------------------------------------------------------:|
| `meshgrid` | ndarray | Meshgrid to be masked. If `None`, uses internally stored meshgrid (created in the `meshgrid` method). |

## Attributes

### `X`

An object on which the transformation will be performed.

**Type:** GeoDataFrame

### `created_meshgrid`

A mesh grid which is a result of transformation.

**Type:** ndarray

