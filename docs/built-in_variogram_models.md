# Built-in Variogram Models

## Intro

$$
d \, - \, \text{distance}
$$

$$
r \, - \, \text{range}
$$

$$
c \, - \, \text{sill}
$$

$$
V(d) \, - \, \text{variogram function}
$$

$$
C(d) \, - \, \text{covariance function}
$$

In general, the covariance function is defined as follows:

$$
C(d) = c - V(d)
$$

For more information about variogram model parameters, please refer to [this tutorial](https://gisgeography.com/semi-variogram-nugget-range-sill/)
or see this [section](creating_custom_variogram_models.md/#parameters-of-the-most-popular-variogram-models) of the 
documentation.

## Gaussian

$$
V(d) = c \cdot \left(1 - \exp\left(-\frac{d^2}{2 \cdot r^2}\right)\right)
$$

$$
C(d) = c \cdot \exp\left(-\frac{d^2}{2 \cdot r^2}\right)
$$

## Exponential

$$
V(d) = c \cdot \left(1 - \exp\left(-\frac{|d|}{r}\right)\right)
$$

$$
C(d) = c \cdot \exp\left(-\frac{|d|}{r}\right)
$$

## Spherical

$$
V(d) = \begin{cases}
        c \cdot \left(1.5 \frac{|d|}{r} - 0.5 \left(\frac{|d|}{r}\right)^3\right) & \text{for } 0 \leq |d| \leq r \\
        c & \text{for } |d| > r
       \end{cases}
$$

$$
C(d) = \begin{cases}
        c \cdot \left(1 - 1.5 \frac{|d|}{r} + 0.5 \left(\frac{|d|}{r}\right)^3\right) & \text{for } 0 \leq |d| \leq r \\
        0 & \text{for } |d| > r
       \end{cases}
$$

## Linear

$$
V(d) = \begin{cases}
        c \cdot \frac{|d|}{r} & \text{for } 0 \leq |d| \leq r \\
        c & \text{for } |d| > r
       \end{cases}
$$


$$
C(d) = \begin{cases}
        c \cdot \left(1 - \frac{|d|}{r}\right) & \text{for } 0 \leq |d| \leq r \\
        0 & \text{for } |d| > r
       \end{cases}
$$
