from abc import abstractmethod


class ParameterInference:
    """
    Abstract class for parameter inference
    """

    @abstractmethod
    def __init__(self, p_obs, *args, **kwargs):
        """
        :param p_obs: (function or method) function/method
        that five the likelihood of an observation
        :param args: (iterable) optional arguments
        :param kwargs: (dictionary) optional arguments
        """
        super().__init__()
        self.p_obs = p_obs

    @abstractmethod
    def update(self, d_idx, resp):
        """
        :param d_idx: (integer) index of the design chosen
        :param resp: (boolean) response of the user
        :return: None
        """

    @abstractmethod
    def get_estimates(self):
        """
        :return: (tuple) estimates of the expected value and (co-)variance
        of the posterior distribution over the parameter space
        """

    @abstractmethod
    def reset(self):
        """
        Reset the prior to the default prior
        :return: None
        """


class ModelComparison:
    """
    Abstract class for model comparison.
    ONLY does the inference part.
    """

    @abstractmethod
    def __init__(self, list_p_obs, *args, **kwargs):
        """
        :param list_p_obs: (iterable) list/tuple of function/method
        that five the likelihood of an observation
        :param args: (iterable) optional arguments
        :param kwargs: (dictionary) optional arguments
        """
        super().__init__()
        self.p_inf = [
            ParameterInference(p) for p in list_p_obs
        ]

    @abstractmethod
    def update(self, d_idx, resp):
        """
        :param d_idx: (integer) index of the design chosen
        :param resp: (boolean) response of the user
        :return: None
        """

    @abstractmethod
    def get_p(self):
        """
        :return: (iterable) list/tuple containing the
        probability for each model
        """

    @abstractmethod
    def reset(self):
        """
        Reset the prior to the default prior
        :return: None
        """


class ADOParameter(ParameterInference):
    """
    Abstract class for adaptive design for parameter inference
    """

    @abstractmethod
    def __init__(self, p_obs, *args, **kwargs):
        super().__init__(p_obs=p_obs, *args, **kwargs)
        self.p_inf = ParameterInference(p_obs=p_obs)

    def get_design(self):
        """
        :return: (int) index of the design
        """


class ADOModel(ModelComparison):
    """
    Abstract class for adaptive design for model comparison
    """

    @abstractmethod
    def __init__(self, p_obs, *args, **kwargs):
        super().__init__(p_obs=p_obs, *args, **kwargs)
        self.p_inf = ParameterInference(p_obs=p_obs)

    def get_design(self):
        """
        :return: (int) index of the design
        """