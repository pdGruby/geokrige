from inspect import isfunction
import numpy as np


class VariogramModel:
    """
    A class for defining variogram model. Variogram & covariance functions are
    defined here.

    Note that
    ---------
    Parameters in variogram and covariance functions must:
    - First parameter must be the distance parameter.
    - Be ordered consistently, i.e., if parameters for the variogram function are
      ordered as 'distance', 'sill', 'range', then parameters for the covariance
      function must be in the same order.

    Attributes
    ----------
    variogram_func : function
        The variogram function.
    covariance_func : function
        The covariance function.
    args : Tuple[str]
        Tuple containing the names of the arguments of the variogram model.
    args_numb : int
        Number of arguments of the variogram model.
    fixed_params : dict
        Dictionary containing fixed parameters of the variogram model.

    Methods
    -------
    set_variogram_func(func)
        Set the variogram function.
    set_covariance_func(func)
        Set the covariance function.
    fix_parameter(param_name, value)
        Fix a parameter of the variogram model to a specific value.
    """

    def __init__(self):
        self.variogram_func = None
        self.covariance_func = None
        self.args = None
        self.args_numb = None
        self.fixed_params = None

    def set_variogram_func(self, func) -> None:
        """
        Set the variogram function.

        Parameters
        ----------
        func : function
            The variogram function.

        Raises
        ------
        TypeError
            If the input for 'func' is not a function object.
        ValueError
            If the provided function has different arguments or argument names
            than the covariance function.

        Returns
        -------
        None
        """
        self.check_passed_function(func)

        args = func.__code__.co_varnames
        args_numb = func.__code__.co_argcount

        if self.args_numb and args_numb != self.args_numb:
            raise ValueError("Invalid input for the 'func' parameter. The function provided has different number "
                             f"of arguments than the covariance function. Variogram function has '{args_numb}'. "
                             f"Covariance function has '{self.args_numb}' arguments.")
        if self.args and self.args != args:
            raise ValueError("Invalid input for the 'func' parameter. The function provided has different argument "
                             f"names than the covariance function. Variogram function arg names: {args}. Covariance "
                             f"function arg names: {self.args}")

        self.variogram_func = np.vectorize(func)
        self.args = args
        self.args_numb = args_numb
        self.fixed_params = {}
        for arg_name in args:
            self.fixed_params[arg_name] = None

    def set_covariance_func(self, func) -> None:
        """
        Set the covariance function.

        Parameters
        ----------
        func : function
            The covariance function.

        Raises
        ------
        TypeError
            If the input for 'variogram_model' is not a function object.
        ValueError
            If the provided function has different arguments or argument names
            than the variogram function.

        Returns
        -------
        None
        """
        self.check_passed_function(func)

        args = func.__code__.co_varnames
        args_numb = func.__code__.co_argcount

        if self.args_numb and args_numb != self.args_numb:
            raise ValueError("Invalid input for the 'func' parameter. The function provided has different number "
                             f"of arguments than the variogram function. Covariance function has '{args_numb}'. "
                             f"Variogram function has '{self.args_numb}' arguments.")
        if self.args and self.args != args:
            raise ValueError("Invalid input for the 'func' parameter. The function provided has different argument "
                             f"names than the variogram function. Covariance function arg names: {args}. Variogram "
                             f"function arg names: {self.args}")

        self.covariance_func = np.vectorize(func)
        self.args = args
        self.args_numb = args_numb
        self.fixed_params = {}
        for arg_name in args:
            self.fixed_params[arg_name] = None

    def fix_parameter(self, param_name, value) -> None:
        """
        Fix a parameter of the variogram model to a specific value.

        Parameters
        ----------
        param_name : str
            The name of the parameter to fix.
        value : float
            The value to fix the parameter to (argument for the
            parameter).

        Raises
        ------
        ValueError
            If the provided parameter name is not valid.

        Returns
        -------
        None
        """
        if param_name not in self.args:
            raise ValueError(f"Invalid input for the 'param_name' parameter: '{param_name}' is not a parameter of "
                             f"the variogram model. Variogram model parameters: {self.args}")

        self.fixed_params[param_name] = value

    @staticmethod
    def check_passed_function(func) -> None:
        if not isfunction(func):
            raise TypeError(f"Invalid input for the 'variogram_model' parameter: expected a function object, got "
                            f"{type(func)} instead")
