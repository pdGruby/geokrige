# VariogramModel

## Description

```py
from geokrige.tools import VariogramModel
```

A class for defining variogram model. Variogram models comprise two key components:

**Variogram Function:**

- this function is fitted to bins of a variogram. Parameters estimated/fixed here are later used by the covariance 
function.

**Covariance Function:**

- this function computes the covariance between points based on their distance. It plays a critical role in assigning 
weights to points with known values during the prediction process.

**Note that parameters in variogram and covariance functions must:**

- First parameter must be the distance parameter.
- Be ordered consistently, i.e., if parameters for the variogram function are ordered as `distance`, `sill`, `range`, 
then parameters for the covariance function must be in the same order.

## Methods

### `set_variogram_func`
**Set the variogram function.**

| Parameter | Accepts  |       Description       |
|:---------:|:--------:|:-----------------------:|
|  `func`   | function | The variogram function. |

### `set_covariance_func`
**Set the covariance function.**

| Parameter | Accepts  |       Description        |
|:---------:|:--------:|:------------------------:|
|  `func`   | function | The covariance function. |

### `fix_parameter`
**Fix a parameter of the variogram model to a specific value.** Later when the `VariogramModel` with fixed parameters is 
passed to the `fit` method, this parameter will not be estimated (it will have a constant value). It can also be set
later in the `fit` method by passing kwargs.

|  Parameter   | Accepts |                           Description                           |
|:------------:|:-------:|:---------------------------------------------------------------:|
| `param_name` |   str   |                The name of the parameter to fix.                |
|   `value`    |  float  | The value to fix the parameter to (argument for the parameter). |

## Attributes

### `variogram_func`

The variogram function.

**Type:** function

### `covariance_func`

The covariance function.

**Type:** function

### `args`

Tuple containing the names of the arguments of the variogram model.

**Type:** Tuple[str]

### `args_numb`

Number of arguments of the variogram model.

**Type:** int

### `fixed_params`

Dictionary containing fixed parameters of the variogram model.

**Type:** dict