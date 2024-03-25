from abc import abstractmethod
from typing import Union, List, Optional, Tuple
import copy

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from scipy.spatial.distance import squareform
from scipy.stats import binned_statistic
from sklearn.metrics import pairwise_distances as pdist
from sklearn.preprocessing import StandardScaler

from geokrige.drafter import KrigingDrafter
from geokrige.tools.variogram_model import VariogramModel
from geokrige.utils.parameters_estimator import ParametersEstimator

from geokrige.functions.variogram import \
    variogram_gaussian_func, variogram_linear_func, \
    variogram_exponential_func, variogram_spherical_func
from geokrige.functions.covariance import \
    covariance_gaussian_func, covariance_linear_func, \
    covariance_exponential_func, covariance_spherical_func


class KrigingBase(KrigingDrafter):
    """
    Base class implementing functionality for the kriging algorithm.

    Attributes
    ----------
    VARIOGRAM_MODEL_FUNCTIONS : dict
        Dictionary containing variogram and covariance functions for different models.

    X : Union[ndarray]
        Input data loaded using the 'load' method. If a GeoDataFrame object has been
        loaded, it undergoes transformation and is then stored in this attribute as
        an ndarray object.

    y : Union[ndarray, str]
        Dependent variable values.

    scaler : StandardScaler
        StandardScaler object for data scaling.

    bins : int
        Number of bins used in variogram calculations.

    distance_metric : str
        Distance metric used for calculations.

    edges : ndarray
        Edges of bins in variogram calculations.

    pairwise_distances : ndarray
        Distance matrix between known data points.

    dissimilarity : ndarray
        Dissimilarity matrix between known data points.

    bins_x : ndarray
        Bins x-values.

    bins_y : ndarray
        Bins y-values.

    variogram_model : VariogramModel
        Variogram model.

    cost_function : str
        Name of the cost function used for variogram model fitting.

    init_args : List[int]
        Initial values for variogram model parameters.

    fixed_params : dict
        Kwargs containing fixed parameters for the variogram model.

    learned_params : List[int]
        Final values of variogram model parameters.

    cov_learned : ndarray
        Learned covariance matrix.

    cov_learned_inv : ndarray
        Inverted covariance matrix.

    dist_matrix : ndarray
        Distance matrix between data points with known value & data
        points for which a value will be estimated.

    Methods
    -------
    load(X, y)
        Load input data into the model.

    variogram(bins, distance_metric, plot)
        Compute and optionally plot the variogram for input data.

    fit(model, cost_function, init_args, plot, **fixed_params)
        Fit a variogram model to input data and optionally plot the
        fitting result.

    predict(X, distance_matrix)
        Predict values for given points.

    evaluate(groups, seed, return_)
        Evaluate model performance using cross-validation.

    summarize(vmin, vmax)
        Plot various aspects of the kriging process: variogram, fitting result
        and covariance matrix.

    plot_variogram(show_fit)
        Plot the variogram.

    plot_cov_matrix(vmin, vmax)
        Plot the covariance matrix.
    """
    VARIOGRAM_MODEL_FUNCTIONS = {
        'gaussian': (variogram_gaussian_func, covariance_gaussian_func),
        'linear': (variogram_linear_func, covariance_linear_func),
        'exp': (variogram_exponential_func, covariance_exponential_func),
        'spherical': (variogram_spherical_func, covariance_spherical_func)
    }

    def __init__(self):
        super().__init__()

        self.X: Optional[Union[np.ndarray, GeoDataFrame, list]] = None
        self.y: Optional[Union[np.ndarray, str, list]] = None
        self.scaler: Optional[StandardScaler] = None

        self.bins: Optional[int] = None
        self.distance_metric: Optional[str] = None
        self.edges: Optional[np.ndarray] = None
        self.pairwise_distances: Optional[np.ndarray] = None
        self.dissimilarity: Optional[np.ndarray] = None
        self.bins_x: Optional[np.ndarray] = None
        self.bins_y: Optional[np.ndarray] = None

        self.variogram_model: Optional[VariogramModel] = None
        self.cost_function: Optional[str] = None
        self.init_args: Optional[List[int]] = None
        self.fixed_params: Optional[dict] = None

        self.learned_params: Optional[list] = None
        self.cov_learned: Optional[np.ndarray] = None
        self.cov_learned_inv: Optional[np.ndarray] = None

        self.dist_matrix: Optional[np.ndarray] = None

        # class workflow flags
        self._data_loaded = False
        self._variogram_created = False
        self._variogram_fitted = False

    def load(self, X: Union[np.ndarray, GeoDataFrame], y: Union[np.ndarray, str]) -> None:
        """
        Load input data into the model.

        Parameters
        ----------
        X : Union[ndarray, GeoDataFrame]
            Input data. Simply put, 2D matrix when it is of type ndarray.
        y : Union[ndarray, str]
            Dependent variable values. When the 'X' parameter is of
            type GeoDataFrame, the 'y' parameter must be a string that
            specifies the column name which stores dependent variable
            in a GeoDataFrame object.

        Returns
        -------
        None
        """
        if not isinstance(X, (np.ndarray, GeoDataFrame)):
            raise TypeError("The 'X' argument must be of type 'ndarray' or 'GeoDataFrame'.")

        if isinstance(X, np.ndarray) and not isinstance(y, np.ndarray):
            raise TypeError("The 'X' argument is of type 'ndarray', so the 'y' argument must be of type 'ndarray'")

        if isinstance(X, GeoDataFrame) and not isinstance(y, str):
            raise TypeError("The 'X' argument is of type 'GeoDataFrame', so the 'y' argument must be of type 'str'.")

        if isinstance(X, GeoDataFrame):
            y = X.loc[:, y].to_numpy()
            X = self._convert_gdf_coords_to_ndarray(X)

        self.X = X
        self.y = y
        self._data_loaded = True

    def variogram(self, bins: int = 20, distance_metric: str = 'euclidean', plot: bool = True) -> None:
        """
        Compute and optionally plot the variogram for input data.

        Parameters
        ----------
        bins : int
            Number of bins used in variogram calculations. Default is 20.
        distance_metric : str
            Distance metric used for calculations. Default is 'euclidean'.
            Valid inputs: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        plot : bool
            Specifies whether to plot the variogram. Default is True.

        Raises
        ------
        ValueError
            - If the 'load' method has not been used yet.
            - If the number of bins provided is less than or equal to 1,
            or greater than the length of the dependent variable vector.
            - If NaN values occur in the variogram y-values due to too many bins.

        Returns
        -------
        None
        """
        if not self._data_loaded:
            raise ValueError(f"Data has not been loaded to the Kriging Class. Please, use the 'load' method first")

        if bins <= 1 or bins > len(self.y):
            raise ValueError(f"Invalid input for the 'bins' parameter: {bins}. The 'bins' argument must be an integer "
                             f"greater than 1 & it must be less or equal to the length of a dependent variable vector "
                             f"-> {len(self.y)}")

        self.scaler, self.y = self._standardize_y(self.y)
        dissimilarity = (np.abs(self.y[:, None] - self.y[None, :]) ** 2) / 2
        pairwise_distances = pdist(self.X, metric=distance_metric)
        pairwise_distances = np.round(pairwise_distances, decimals=10)
        # in some cases, there might occur a computational problem when pairwise_distances is not rounded - the matrix
        # will not be symmetric

        bins_y, edges, _ = binned_statistic(squareform(pairwise_distances), squareform(dissimilarity),
                                            statistic='mean', bins=bins)
        bins_x = edges[:-1] + (edges[1] - edges[0]) / 2

        if np.isnan(bins_y).any():
            raise ValueError(f"Too many bins were created, resulting in None values occurrence in the variogram "
                             f"y-values. Specify fewer bins in the 'bins' argument. Found "
                             f"{np.isnan(bins_y).sum()} None values in the variogram y-values.")

        self.bins = bins
        self.distance_metric = distance_metric
        self.edges = edges
        self.pairwise_distances = pairwise_distances
        self.dissimilarity = dissimilarity
        self.bins_x = bins_x
        self.bins_y = bins_y

        self._variogram_created = True
        if plot:
            self.plot_variogram(show_fit=False)

    def fit(self, model: Union[VariogramModel, str] = 'gaussian', cost_function: str = 'soft_l1',
            init_args: List[int] = None, plot: bool = True, **fixed_params: float) -> None:
        """
        Fit a variogram model to input data and optionally plot the fitting result.

        Parameters
        ----------
        model : Union[VariogramModel, str]
            User's defined variogram model or a name of the model to use. Default
            is 'gaussian'. Valid inputs: [VariogramModel object, 'gaussian', 'exp',
            'spherical', 'linear']
        cost_function : str
            Cost function used during model fitting. Default is 'soft_l1'. Valid
            inputs: ['linear', 'soft_l1', 'huber', 'cauchy', 'arctan']
        init_args : List[int], optional
            Initial values for variogram model parameters. This is a starting
            point in the optimization process.
        plot : bool, optional
            Specifies whether to plot a fitting result. Default is True.
        **fixed_params : kwargs with floats, optional
            Fixed parameters for the variogram model. The keywords must have the
            same name as the variogram model parameters (built-in variogram models
            have 'distance', 'range_param', 'sill_param').

        Raises
        ------
        ValueError
            - If the 'variogram' method has not been used yet.
            - If the 'init_args' argument has more initial values than the number
            of variogram model parameters.

        Returns
        -------
        None
        """
        if not self._variogram_created:
            raise ValueError(f"A variogram has not been created yet. Please, use the 'variogram' method first")

        if model not in self.VARIOGRAM_MODEL_FUNCTIONS.keys() \
                and not isinstance(model, VariogramModel):
            raise TypeError(f"Invalid input for the 'model' parameter: expected 'VariogramModel' object "
                            f"or one of listed strings ({self.VARIOGRAM_MODEL_FUNCTIONS.keys()}), "
                            f"got {model} instead.")

        if not isinstance(init_args, (list, type(None))):
            raise TypeError(f"Invalid input for the 'init_args' parameter: should be a list or None, got "
                            f"{type(init_args)} instead")

        if isinstance(model, str):
            variogram_func, covariance_func = self.VARIOGRAM_MODEL_FUNCTIONS[model]
            model = VariogramModel()
            model.set_variogram_func(variogram_func)
            model.set_covariance_func(covariance_func)
        model = copy.deepcopy(model)

        self.variogram_model = model
        self.cost_function = cost_function
        self.init_args = init_args
        self.fixed_params = fixed_params

        if init_args is None:
            init_args = [1] * (self.variogram_model.args_numb - 1)
        if len(init_args) != (self.variogram_model.args_numb - 1):
            raise ValueError(f"Invalid input for the 'init_args' parameter: variogram model has "
                             f"{self.variogram_model.args_numb - 1} potential parameters to estimate, but "
                             f"{len(init_args)} init values were provided")

        # Parameters estimation
        for key, value in fixed_params.items():
            self.variogram_model.fix_parameter(key, value)

        weighted_values = np.histogram(squareform(self.pairwise_distances), bins=self.edges)[0].astype(float)
        estimator = ParametersEstimator(model)
        learned_params = estimator.estimate_parameters(self.bins_x, *init_args,
                                                       bins_y=self.bins_y,
                                                       weighted_values=weighted_values,
                                                       cost_function=cost_function)
        self.learned_params = learned_params

        # Covariance matrix creation
        cov_learned = self._create_cov_matrix()
        cov_learned_inv = np.linalg.pinv(cov_learned)

        self.cov_learned = cov_learned
        self.cov_learned_inv = cov_learned_inv

        self._variogram_fitted = True
        if plot:
            self.plot_variogram(show_fit=True)

    def predict(self, X: Union[List[np.ndarray], np.ndarray], distance_matrix: np.ndarray = None) -> np.ndarray:
        """
        Predict values for given points.

        Parameters
        ----------
        X : Union[List[ndarray], ndarray]
            Input data. List of matrices (numpy.meshgrid), or 2D matrix
            (numpy.ndarray).
        distance_matrix : ndarray, optional
            Distance matrix between known data points & unknown data points.
            This parameter can be used when the same distance matrix is used
            in many models (optimization feature).

        Raises
        ------
        ValueError
            - If the 'fit' method has not been used yet.
            - If the number of dimensions (coordinates) provided for prediction
            does not match the number of dimensions the model was trained on.

        Returns
        -------
        ndarray
            Predicted values.
        """
        if not self._variogram_fitted:
            raise ValueError(f"A variogram function has not been fitted yet. Please, use the 'fit' method first")

        if isinstance(X, list):  # np.meshgrid
            shape = tuple(X[0].shape)
            X = np.column_stack([matrix.ravel() for matrix in X])
        elif isinstance(X, np.ndarray):  # 2D matrix
            shape = X.shape[0]
        else:
            raise TypeError(f"The type of the 'X' argument is invalid! Valid types are [numpy.ndarray, list], "
                            f"got {type(X)} instead")

        if X.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"The number of dimensions (coordinates) provided for prediction does not match the number "
                f"of dimensions the model was trained on. Training data dimensions: {self.X.shape[1]}. "
                f"Prediction data dimensions: {X.shape[1]}")

        if distance_matrix is None:
            self.dist_matrix = pdist(self.X, X, metric=self.distance_metric)
        else:
            self.dist_matrix = distance_matrix

        predicted_values = self._predict_values()
        return self.scaler.inverse_transform(predicted_values.reshape(1, -1)).reshape(shape)

    def evaluate(self, groups: int = 5, seed: int = None, return_: bool = False) -> Optional[dict]:
        """
        Evaluate model performance using cross-validation.

        Parameters
        ----------
        groups : int
            Number of groups for cross-validation. Default is 5.
        seed : int, optional
            Seed for random number generator.
        return_ : bool
            Specifies whether to return evaluation results. Default is False.

        Raises
        ------
        ValueError
            - If the 'fit' method has not been used yet.
            - If the number of 'groups' provided is less than or equal to 1,
            or greater than the number of the points on which the model was
            trained on.

        Returns
        -------
        Optional[dict]
            Evaluation results in dictionary form if return_ is True.
        """
        if not self._variogram_fitted:
            raise ValueError(f"A variogram function has not been fitted yet. Please, use the 'fit' method first")

        if groups <= 1 or groups > self.X.shape[0]:
            raise ValueError(f"Invalid input for the 'groups' parameter: {groups}. The 'groups' argument must be an "
                             f"integer greater than 1 & must not be greater than the number of the points on which "
                             f"the model was trained on -> {self.X.shape[0]}")

        if seed:
            np.random.seed(seed)

        evaluator = copy.deepcopy(self)
        data = np.column_stack([self.X, self.scaler.inverse_transform(self.y.reshape(1, -1)).ravel()])
        np.random.shuffle(data)

        mae_result = 0
        rmse_result = 0
        testing_groups = np.array_split(data, groups)
        for group in testing_groups:
            mask = np.all(np.isin(data, group), axis=1)

            train_X = data[~mask][:, 0: data.shape[1] - 1]
            train_y = data[~mask][:, -1]

            test_X = data[mask][:, 0: data.shape[1] - 1]
            test_y = data[mask][:, -1]

            # Simulating that the model was trained without testing points ---------------------------------------------
            pairwise_distances = pdist(train_X, metric=self.distance_metric)
            pairwise_distances = np.round(pairwise_distances, decimals=10)
            evaluator.pairwise_distances = pairwise_distances

            cov_learned = evaluator._create_cov_matrix()
            cov_learned_inv = np.linalg.pinv(cov_learned)
            evaluator.cov_learned = cov_learned
            evaluator.cov_learned_inv = cov_learned_inv

            dist_matrix = pdist(train_X, test_X, metric=self.distance_metric)
            evaluator.y = evaluator.scaler.transform(train_y.reshape(-1, 1)).ravel()
            # ----------------------------------------------------------------------------------------------------------

            predicted_y = evaluator.predict(test_X, dist_matrix)

            mae = np.abs(predicted_y - test_y).mean()
            rmse = np.sqrt(((predicted_y - test_y) ** 2).mean())
            mae_result += mae
            rmse_result += rmse

        if return_:
            return {'MAE': mae_result / groups, 'RMSE': rmse_result / groups}
        print(f'Mean Absolute Error: {mae_result / groups}')
        print(f'Root-Mean-Square Error: {rmse_result / groups}')

    def _create_cov_matrix(self) -> np.ndarray:
        """
        Create a covariance matrix based on learned variogram model parameters.
        Can be overridden if a kriging method requires different covariance matrix.

        Returns
        -------
        ndarray
            Created covariance matrix.
        """
        return self.variogram_model.covariance_func(self.pairwise_distances, *self.learned_params)

    @abstractmethod
    def _predict_values(self) -> np.ndarray:
        """
        Abstract method for predicting values. Must be implemented by concrete subclasses.

        Returns
        -------
        ndarray
            Predicted values.
        """
        raise NotImplementedError

    @staticmethod
    def _standardize_y(y) -> Tuple[StandardScaler, np.ndarray]:
        """
        Standardize dependent variable values.

        Parameters
        ----------
        y : ndarray
            Dependent variable values.

        Returns
        -------
        Tuple[StandardScaler, ndarray]
            Tuple containing StandardScaler object and standardized dependent variable values.
        """
        scaler = StandardScaler()
        scaler.fit(y.reshape(-1, 1))
        return scaler, scaler.transform(y.reshape(-1, 1)).ravel()

    @staticmethod
    def _convert_gdf_coords_to_ndarray(X: GeoDataFrame) -> np.ndarray:
        """
        Converts GeoDataFrame coordinates to numpy array.

        Parameters
        ----------
        X : GeoDataFrame
            Input GeoDataFrame.

        Returns
        -------
        np.ndarray
        """
        coordinates = X.get_coordinates(include_z=True)
        X = np.column_stack([coordinates[column] for column in coordinates if not coordinates[column].isna().all()])

        return X
