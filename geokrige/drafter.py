import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import squareform


class KrigingDrafter:
    pairwise_distances = None
    dissimilarity = None
    bins_x = None
    bins_y = None

    variogram_model = None
    cost_function = None
    learned_params = None
    cov_learned = None

    # class workflow flags
    _data_loaded: bool
    _variogram_created: bool
    _variogram_fitted: bool

    def summarize(self, vmin: float = None, vmax: float = None) -> None:
        """
        Plot various aspects of the kriging process: variogram, fitting
        result and covariance matrix.

        Parameters
        ----------
        vmin : float, optional
            Minimum value for color normalization in covariance matrix plot.
        vmax : float, optional
            Maximum value for color normalization in covariance matrix plot.

        Returns
        -------
        None
        """
        if not self._variogram_fitted:
            raise ValueError(f"A variogram function has not been fitted yet. Please, use the 'fit' method first")

        fig = plt.figure(figsize=(20, 6))
        gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[2, 2], wspace=0.1, hspace=0.1)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[:, 1])

        title_size = 12
        axis_label_size = 10
        ticks_label_size = 8
        hide_tick_params = {'labelsize': 0, 'length': 0, 'colors': 'white'}

        # Variograms
        ax1.scatter(squareform(self.pairwise_distances), squareform(self.dissimilarity), color='red', marker='.', s=1,
                    label='Raw variogram')
        ax1.scatter(self.bins_x, self.bins_y, color='black', marker='x', s=100,
                    label='Binned variogram')
        ax1.set_title('Variogram', fontweight='bold', size=title_size)

        ax2.scatter(self.bins_x, self.bins_y, color='black', marker='x', s=100,
                    label='Binned variogram')
        ax2.plot(self.bins_x,
                 self.variogram_model.variogram_func(self.bins_x, *self.learned_params),
                 "--",
                 label=self.cost_function)
        ax2.set_xlabel('Distance', fontweight='bold', size=axis_label_size)

        for ax in [ax1, ax2]:
            ax.set_ylabel('Dissimilarity', fontweight='bold', size=axis_label_size)
            ax.tick_params(axis='both', which='major', labelsize=ticks_label_size)
            ax.legend(prop={'size': ticks_label_size})

        ax1.tick_params(axis='x', which='major', **hide_tick_params)

        # Covariance matrix
        im = ax3.imshow(self.cov_learned, vmin=vmin, vmax=vmax)
        ax3.set_title("Covariance Matrix", fontweight='bold', size=title_size)
        ax3.set_xlabel(f"Covariance condition number: {np.linalg.cond(self.cov_learned).round(2)}",
                       size=axis_label_size)
        ax3.tick_params(axis='both', which='major', **hide_tick_params)

        cbar = fig.colorbar(im, location='bottom', fraction=0.04)
        cbar.ax.tick_params(labelsize=ticks_label_size)

    def plot_variogram(self, show_fit: bool = False):
        """
        Plot the variogram.

        Parameters
        ----------
        show_fit : bool
            Specifies whether to plot the fitted variogram or a raw variogram.
            Default is False.

        Raises
        ------
        ValueError
            - If the 'variogram' method has not been used yet.
            - If the 'fit' method has not been used yet and the 'show_fit' parameter
            is set to True.

        Returns
        -------
        None
        """
        if not self._variogram_created:
            raise ValueError(f"A variogram has not been created yet. Please, use the 'variogram' method first")
        if show_fit and not self._variogram_fitted:
            raise ValueError(f"A variogram function has not been fitted yet. Please, use the 'fit' method first")

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.set_title('Variogram', fontweight='bold')
        ax.set_xlabel('Distance', fontweight='bold')
        ax.set_ylabel('Dissimilarity', fontweight='bold')

        if show_fit:
            ax.scatter(self.bins_x, self.bins_y, color='black',
                       marker='x', s=100, label='Binned variogram')
            x = self.bins_x.tolist()
            ax.plot(x,
                    self.variogram_model.variogram_func(x, *self.learned_params),
                    "--",
                    label=self.cost_function)
        else:
            ax.scatter(squareform(self.pairwise_distances), squareform(self.dissimilarity),
                       color='red', marker='.', s=1, label='Raw variogram')
            ax.scatter(self.bins_x, self.bins_y, color='black', marker='x', s=100,
                       label='Binned variogram')

        ax.legend()

    def plot_cov_matrix(self, vmin: float = None, vmax: float = None):
        """
        Plot the covariance matrix.

        Parameters
        ----------
        vmin : float, optional
            Minimum value for color normalization in covariance matrix plot.
        vmax : float, optional
            Maximum value for color normalization in covariance matrix plot.

        Raises
        ------
        ValueError
            - If the 'fit' method has not been used yet.

        Returns
        -------
        None
        """
        if not self._variogram_fitted:
            raise ValueError(f"Variogram function has not been fitted yet. Please, use the 'fit' method first")

        if vmin is None:
            vmin = 0

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        im = ax.imshow(self.cov_learned, vmin=vmin, vmax=vmax)
        ax.set_title("Covariance Matrix")
        ax.set_xlabel(f"Covariance condition number: {np.linalg.cond(self.cov_learned).round(2)}")
        fig.colorbar(im)
