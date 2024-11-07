from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
import numpy as np

class TukeyFence(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Tukey Fence method for outlier detection.
    The Tukey Fence method is a simple method for detecting outliers in a dataset.
    It is based on the interquartile range (IQR) of the dataset.
    The IQR is defined as the difference between the 75th percentile and the 25th percentile of the dataset.
    The Tukey Fence method defines an outlier as any value that is below the 25th percentile minus 1.5 times the IQR
    or above the 75th percentile plus 1.5 times the IQR.
    Parameters
    ----------
    iqr_multiplier : float, optional (default=1.5)
        The multiplier for the IQR in the Tukey Fence method.
    lower_bound : float, optional (default=None)
        The lower bound for the Tukey Fence method. If None, the lower bound will be set to the 25th percentile
        minus the IQR times the multiplier.
    upper_bound : float, optional (default=None)
        The upper bound for the Tukey Fence method. If None, the upper bound will be set to the 75th percentile
        plus the IQR times the multiplier.
    Attributes
    ----------
    lower_bound : float
        The lower bound for the Tukey Fence method.
    upper_bound : float
        The upper bound for the Tukey Fence method.
    Examples
    --------
    >>> import pandas as pd
    >>> from nsoml.preprocessing import TukeyFence
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    >>> tukey = TukeyFence()
    >>> tukey.fit(data)
    >>> tukey.lower_bound
    0.5
    >>> tukey.upper_bound
    10.5
    >>> tukey.transform(data)
       A
    0  1
    1  2
    2  3
    3  4
    4  5
    5  6
    6  7
    7  8
    8  9
    9  10
    """
    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self.bounds = np.array([])

    def fit(self, X, y=None, **fit_params):
        """Fit the Tukey Fence method to the data."""
        self.bounds = np.zeros((X.shape[1], 2))

        # Replace IQR if provided
        if 'iqr_multiplier' in fit_params:
            self.iqr_multiplier = fit_params['iqr_multiplier']

        for i in range(X.shape[1]):
            q1 = np.percentile(X[:, i], 25)
            q3 = np.percentile(X[:, i], 75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            self.bounds[i] = [lower_bound, upper_bound]
        return self
    
    def transform(self, X):
        """Transform the data using the Tukey Fence method."""
        for i in range(X.shape[1]):
            lower_bound = self.bounds[i, 0]
            upper_bound = self.bounds[i, 1]
            X[X[:, i] < lower_bound, i] = lower_bound
            X[X[:, i] > upper_bound, i] = upper_bound
        return X


    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform the data using the Tukey Fence method."""
        return self.fit(X, y, **fit_params).transform(X)


                




