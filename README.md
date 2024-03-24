<p align="center">
    <img alt="GeoKrige banner" src="docs/images/geokrige_baner.png"/>
</p>

For comprehensive documentation, please visit the [readthedocs webpage](https://geokrige.readthedocs.io/latest/).

[![PyPi downloads](https://static.pepy.tech/badge/geokrige)](https://pepy.tech/project/cloupy)

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

## Support

Please, use the [`Q&A section`](https://github.com/pdGruby/geokrige/discussions/categories/q-a) in case you need a help.
