from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

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
    def __init__(self, iqr_multiplier: float = 1.5, lower_bound=None, upper_bound=None):
        self.lower_bound = lower_bound 
        self.upper_bound = upper_bound
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        """Fit the Tukey Fence method to the data."""
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        if self.lower_bound is None:
            self.lower_bound = q1 - self.iqr_multiplier * iqr
        if self.upper_bound is None:
            self.upper_bound = q3 + self.iqr_multiplier * iqr
        return self
    
    def transform(self, X):
        """Transform the data using the Tukey Fence method."""
        return X[(X >= self.lower_bound) & (X <= self.upper_bound)]

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform the data using the Tukey Fence method."""
        return self.fit(X).transform(X)


                




