# OrdinaryKriging, SimpleKriging, UniversalKriging

## Description

```py
from geokrige.methods import OrdinaryKriging
from geokrige.methods import SimpleKriging
from geokrige.methods import UniversalKriging
```

These classes represent various implementations of Kriging Methods. They all inherit from the same base class, which 
provides all necessary functionalities for users. From a user's perspective, there are no visible changes when importing 
a specific class. Technically, **importing a specific class alters the method by which the covariance matrix is created 
and/or the prediction is performed.**

### OrdinaryKriging
In Ordinary Kriging, it is assumed that the **variable values are spatially stable, without trends**. This method 
estimates  the value of an unknown point by considering **the weights of neighboring points.** The most fundamental 
Kriging Method.

### SimpleKriging
Simple Kriging, like Ordinary Kriging, **assumes spatial stability without trends.** However, it incorporates the mean 
value of the variable as a reference point. **The differences between observed values and this mean, known as residuals, 
are used to calculate weights for interpolation.** These residuals represent the spatial variability beyond the mean and 
determine the influence of each neighboring point on the interpolation of the unknown point.

### UniversalKriging
Universal Kriging builds upon Ordinary Kriging by **allowing for the inclusion of explanatory variables that may 
introduce trends or non-stationarity in the variable.** It extends the capabilities of Kriging by incorporating trend 
functions into the interpolation model. **This Kriging Method is not supported yet by the GeoKrige package.**

## Methods

### `load`
**Load input data into the model.** The `X` parameter accepts 2D matrices or GeoDataFrame objects as input. The `y`
parameter is a vector of the same length as the input matrix or a string specifying the column name from the GeoDataFrame 
object storing values for interpolation.

| Parameter |        Accepts        |                                                                                         Description                                                                                         |
|:---------:|:---------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    `X`    | ndarray, GeoDataFrame |                                                                Input data. Simply put, 2D matrix when it is of type ndarray.                                                                |
|    `y`    |     ndarray, str      | Dependent variable values. When the `X` parameter is of type GeoDataFrame, the `y` parameter must be a string with the column name that stores dependent variable in a GeoDataFrame object. |

### `variogram`
**Compute and optionally plot the variogram for input data.** The `bins` parameter must be a value less than the 
length of a dependent variable vector. There is no universal rule for the perfect number of bins, however in most cases 
10-30 range will be appropriate.

|     Parameter     | Accepts |                                                                     Description                                                                     |
|:-----------------:|:-------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|
|      `bins`       |   int   |                                           Number of bins used in variogram calculations. Default is `20`.                                           |
| `distance_metric` |   str   | Distance metric used for calculations. Default is `euclidean`.<br><br>**Valid inputs:** `cityblock`, `cosine`, `euclidean`, `l1`, `l2`, `manhattan` |
|      `plot`       |  bool   |                                             Specifies whether to plot the variogram. Default is `True`.                                             |

### `fit`
**Fit the variogram model to the input data and optionally plot a fitting result.** If the variogram model parameters 
are not fixed by user, the [`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) 
optimization method is used to find parameters. The optimization process can be slightly adjusted with the usage of 
`cost_function` & `init_args` parameters. If only one parameter of the variogram model is fixed, the other one will be 
estimated.

|    Parameter     |       Accepts       |                                                                                             Description                                                                                              |
|:----------------:|:-------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     `model`      | str, VariogramModel |           User's defined variogram model or a name of the model to use. Default is `gaussian`.<br><br>**Valid inputs:** `VariogramModel object`, `gaussian`, `exp`, `spherical`, `linear`            |
| `cost_function`  |         str         |                               Cost function used during model fitting. Default is `soft_l1`.<br><br>**Valid inputs:** `linear`, `soft_l1`, `huber`, `cauchy`, `arctan`                               |
|   `init_args`    |        list         |                                            Initial values for variogram model parameters. This is a starting point in the optimization process. Optional.                                            |
|      `plot`      |        bool         |                                                                    Specifies whether to plot a fitting result. Default is `True`.                                                                    |
| `**fixed_params` | kwargs with floats  | Fixed parameters for the variogram model. The keywords must have the same name as the variogram model parameters (built-in variogram models have `distance`, `range_param`, `sill_param`). Optional. |

There is a separate section in the documentation dedicated to variogram models. **More information about built-in 
variogram models & defining custom variogram models can be found [here](creating_custom_variogram_models.md).**

### `predict`
**Predict values for given points.** The method is designed to accept mesh grids created by [`numpy.meshgrid`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html) 
function, but it also accepts 2D `numpy.ndarray`.

In fact, the `numpy.meshgrid` function returns a list of nested `numpy.ndarray` objects, so when the method 
receives a list as an input, it assumes that a mesh grid has been passed. 

If an object of type `numpy.ndarray` is passed, then it must be a 2D matrix with coordinates specified in the columns 
(one column for x-coordinates, one column for y-coordinates etc.). 

When a mesh grid is passed as an input, the method returns a matrix with the same shape as matrices of a mesh grid. When 
a 2D matrix is passed, the method returns a vector of the same length as the input matrix.

|     Parameter     |        Accepts         |                                                                                    Description                                                                                     |
|:-----------------:|:----------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|        `X`        | List[ndarray], ndarray |                                                  Input data. List of matrices (`numpy.meshgrid`), or 2D matrix (`numpy.ndarray`).                                                  |
| `distance_matrix` |        ndarray         | Distance matrix between known data points & unknown data points. This parameter can be used when the same distance matrix is used in many models (optimization feature). Optional. |

**Note that:** the GeoKrige package supports a transformation of `GeoDataFrame` objects & `rasterio` objects into mesh 
grids. For more information, please see [GDFTransformer](./transformer_gdf_class_description.md) & 
[RasterioTransformer](./transformer_rasterio_class_description.md) sections.

### `evaluate`
**Evaluate model performance using cross-validation.** By default, the MAE & RMSE score are printed out. However, the
results can be returned in a dictionary form.

| Parameter | Accepts |                             Description                             |
|:---------:|:-------:|:-------------------------------------------------------------------:|
| `groups`  |   int   |       Number of groups for cross-validation. Default is `5`.        |
|  `seed`   |   int   |             Seed for random number generator. Optional.             |
| `return_` |  bool   | Specifies whether to return evaluation results. Default is `False`. |

### `summarize`
**Plot various aspects of the kriging process: variogram, fitting result and covariance matrix.**

| Parameter | Accepts |                                Description                                 |
|:---------:|:-------:|:--------------------------------------------------------------------------:|
|  `vmin`   |  float  | Minimum value for color normalization in covariance matrix plot. Optional. |
|  `vmax`   |  float  | Maximum value for color normalization in covariance matrix plot. Optional. |

### `plot_variogram`
**Plot the variogram.** By default, the variogram without a fitting is plotted.

| Parameter  | Accepts |                                      Description                                       |
|:----------:|:-------:|:--------------------------------------------------------------------------------------:|
| `show_fit` |  bool   | Specifies whether to plot the fitted variogram or a raw variogram. Default is `False`. |

### `plot_cov_matrix`
Plot the covariance matrix.

| Parameter | Accepts |                                Description                                 |
|:---------:|:-------:|:--------------------------------------------------------------------------:|
|  `vmin`   |  float  | Minimum value for color normalization in covariance matrix plot. Optional. |
|  `vmax`   |  float  | Maximum value for color normalization in covariance matrix plot. Optional. |

## Attributes

### `VARIOGRAM_MODEL_FUNCTIONS`

Dictionary containing variogram and covariance functions for different models.

**Type:** dict

### `X_loaded`

The original `X` data loaded using the `load` method (no transformation performed on this object).

**Type:** ndarray, GeoDataFrame

### `y_loaded`

The original `y` data loaded using the `load` method (no transformation performed on this object).

**Type:** ndarray, str

### `X`

Input data loaded using the `load` method. If a `GeoDataFrame` object has been loaded, it undergoes transformation and 
is then stored in this attribute as an `ndarray` object.

**Type:** ndarray

### `y`

Dependent variable values. The values here are transformed by the `scaler` object.

**Type:** ndarray, str

### `scaler` 

StandardScaler object for data scaling.

**Type:** [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

### `bins`

Number of bins used in variogram calculations.

**Type:** int

### `distance_metric`

Distance metric used for calculations.

**Type:** str

### `edges`

Edges of bins in variogram calculations.

**Type:** ndarray

### `pairwise_distances`

Distance matrix between known data points.

**Type:** ndarray

### `dissimilarity`

Dissimilarity matrix between known data points.

**Type:** ndarray

### `bins_x`

Bins x-values.

**Type:** ndarray

### `bins_y`

Bins y-values.

**Type:** ndarray

### `variogram_model`

Variogram model.

**Type:** VariogramModel

### `cost_function`

Name of the cost function used for variogram model fitting.

**Type:** str

### `init_args`

Initial values for variogram model parameters.

**Type:** List[int]

### `fixed_params`

Kwargs containing fixed parameters for the variogram model.

**Type:** dict

### `learned_params`

Final values of variogram model parameters.

**Type:** List[int]

### `cov_learned`

Learned covariance matrix.

**Type:** ndarray

### `cov_learned_inv`

Inverted covariance matrix.

**Type:** ndarray

### `dist_matrix`

Distance matrix between data points with known value & data points for which a value will be estimated.

**Type:** ndarray
