<p align="center">
  <img alt="GeoKrige banner" src="https://geokrige.readthedocs.io/latest/images/geokrige_baner.png"/>
</p>

For comprehensive documentation, please visit the [readthedocs webpage](https://geokrige.readthedocs.io/latest/).

[![PyPi Downloads](https://static.pepy.tech/badge/geokrige)](https://pepy.tech/project/geokrige)
[![Documentation Status](https://readthedocs.org/projects/geokrige/badge/?version=latest)](https://geokrige.readthedocs.io/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/776209752.svg)](https://zenodo.org/doi/10.5281/zenodo.10866997)

## What is GeoKrige?

**GeoKrige is a Python package designed for spatial interpolation using Kriging Methods.** While primarily tailored for 
geospatial analysis, it is equally applicable to other spatial analysis tasks.

**GeoKrige** simplifies kriging interpolation, offering an intuitive interface akin to the `SciKit-learn` package.

## Key Features of GeoKrige

- Seamless integration with the [`GeoPandas`](https://geopandas.org/en/stable/#) and [`rasterio`](https://rasterio.readthedocs.io/en/stable/) packages
- **Generation of interpolated mesh grids aligned with the boundaries of provided shapefiles** (ideal for creating 
interpolation maps)
- Evaluation tool for created kriging models
- **Support for multidimensional interpolation** (Multidimensional Kriging)
- Several default variogram models, flexibility for users to define custom models

## Contribution

There are few topics to which you can contribute:

- implementing the Universal Kriging method
- creating unit tests
- designing a cool-looking logo :)

If you have other ideas on how to improve the package, feel free to share them in the [`Ideas sction`](https://github.com/pdGruby/geokrige/discussions/categories/ideas)

When writing a commit message, please adhere to the guidelines outlined in [this tutorial](https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/).

**The commit message template:**

- `[<type>/...] <commit message>`

The commit types are described in the linked tutorial, but try to adhere to: `feat`, `bugfix`, `docs`, `refact`, `test`, 
`other`. If a specific commit pertains to multiple types, separate them with `/` and ensure they are ordered 
alphabetically. Keep the commit message title brief and descriptive. If a longer description is necessary, please use 
the second `-m` option.

**Exemplary commit messages:**

- `[bugfix] Fix the 'evaluate' method`

- `[docs/refact] Change the 'KrigingBase' class attributes & adjust the documentation to these changes`

## Support

Please, use the [`Q&A section`](https://github.com/pdGruby/geokrige/discussions/categories/q-a) in case you need a help.
