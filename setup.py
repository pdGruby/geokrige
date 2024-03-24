from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setup(
    name="geokrige",
    version='{{VERSION_PLACEHOLDER}}',
    author="Kamil Grala",
    author_email="grala.kamil@outlook.com",
    description="GeoKrige is a Python package designed for spatial interpolation using Kriging Methods. While "
                "primarily tailored for geospatial analysis, it is equally applicable to other spatial analysis tasks.",
    url="https://github.com/pdGruby/geokrige",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[line.strip() for line in open(os.path.join(here, "requirements.txt"), "r").readlines()],
    keywords=['python', 'interpolation', 'spatial-analysis', 'kriging',
              'geospatial-analysis', 'interpolation-methods', 'geospatial-visualization'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: GIS"
    ],
    license='GPL-3.0'
)
