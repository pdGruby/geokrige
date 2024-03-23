from scipy import optimize
import numpy as np
from geokrige.tools.variogram_model import VariogramModel


class ParametersEstimator:
    """
    Estimates parameters for a variogram model based on given variogram values.

    Parameters
    ----------
    variogram_model : VariogramModel
        The variogram model for which parameters are to be estimated.

    Methods
    -------
    estimate_parameters((init_args, bins_y, weighted_values, cost_function)
        Estimates parameters for the variogram model.

    calc_residuals(*args, variogram_model, bins_y, weighted_values)
        Calculates residuals between observed and predicted variogram values.
    """
    def __init__(self, variogram_model: VariogramModel):
        self.variogram_model: VariogramModel = variogram_model

    def estimate_parameters(self, *init_args: np.ndarray, bins_y: np.ndarray = None,
                            weighted_values: np.ndarray = None, cost_function: str = None) -> np.ndarray:
        """
        Estimate parameters for the variogram model.

        Parameters
        ----------
        init_args : np.ndarray
            Initial values for the parameters to be estimated.
        bins_y : np.ndarray, optional
            Binned variogram values.
        weighted_values : np.ndarray, optional
            Weighted values for the optimization.
        cost_function : str, optional
            Name of the cost/loss function used for optimization.

        Returns
        -------
        np.ndarray
            Estimated parameters for the variogram model.

        Raises
        ------
        ValueError
            - If the input for the 'variogram_model' parameter is not an
            instance of VariogramModel.
            - If the number of initialization arguments does not match the number
            of parameters in the variogram model.
        """
        if not isinstance(self.variogram_model, VariogramModel):
            raise ValueError(
                f"Invalid input for the 'variogram_model' parameter: expected {VariogramModel} object, got "
                f"{type(self.variogram_model)} instead")

        if len([*init_args]) != self.variogram_model.args_numb:
            raise ValueError(
                f"Function takes exactly {self.variogram_model.args_numb} parameters, "
                f"but {len([*init_args])} values were provided for the init arguments")

        first_arg_name = self.variogram_model.args[0]
        self.variogram_model.fix_parameter(first_arg_name, [*init_args][0])
        params_to_estimate = [*init_args]

        indexes_to_del = []
        for key, value in self.variogram_model.fixed_params.items():
            if value is None:
                continue
            index = self.variogram_model.args.index(key)
            indexes_to_del.append(index)

        for index in sorted(indexes_to_del, reverse=True):
            del params_to_estimate[index]

        if len(params_to_estimate) == 0:
            parameters = [self.variogram_model.fixed_params[arg_name] for arg_name in self.variogram_model.args]
            return np.array(parameters[1:])  # do not return the first parameter - it differs in each step

        estimated_parameters = optimize.least_squares(
            self.calc_residuals, [*params_to_estimate],
            bounds=([0.0] * len(params_to_estimate), [np.inf] * len(params_to_estimate)),
            loss=cost_function,
            kwargs={'variogram_model': self.variogram_model, 'bins_y': bins_y,
                    'weighted_values': weighted_values}
        )

        parameters_to_return = estimated_parameters.x.tolist()
        for key, value in self.variogram_model.fixed_params.items():
            if value is None:
                continue
            index = self.variogram_model.args.index(key)
            parameters_to_return.insert(index, value)

        # do not return the first parameter (distance/lag value) that differs in each step (depending on whether we
        # are estimating parameters or calculating unknown values, e.g. in the 'predict' method)
        return parameters_to_return[1:]

    @staticmethod
    def calc_residuals(*args: np.ndarray, variogram_model: VariogramModel, bins_y: np.ndarray,
                       weighted_values: np.ndarray) -> np.ndarray:
        """
        Calculate residuals between observed and predicted variogram values.

        Parameters
        ----------
        args : tuple of np.ndarray
            Parameters for the variogram model.
        variogram_model : VariogramModel
            The variogram model.
        bins_y : np.ndarray
            Observed variogram values.
        weighted_values : np.ndarray
            Weighted values for the optimization.

        Returns
        -------
        np.ndarray
            Residuals between observed and predicted variogram values.
        """
        parameters = [*args][0].tolist()

        for key, value in variogram_model.fixed_params.items():
            if value is None:
                continue
            index = variogram_model.args.index(key)
            parameters.insert(index, value)

        residuals = variogram_model.variogram_func(*parameters) - bins_y
        return residuals * weighted_values
