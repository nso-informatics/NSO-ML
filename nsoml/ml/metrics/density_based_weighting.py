from typing import List, Union
from sklearn.utils._param_validation import validate_params, StrOptions
from sklearn.metrics import f1_score
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd

class DensityBasedWeighting(object):
    """Density-based weighting for imbalanced datasets.
    This class provides static methods to compute density-based weighting KDE for imbalanced datasets.
    """
    kde = None
    average_density = None
    density_weights = None
    n = None

    @staticmethod
    def compute_density_weights(y_true: Union[List, np.ndarray]) -> np.ndarray:
        """Compute density-based weights for imbalanced datasets.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth target values.

        Returns
        -------
        density_weights : array-like of shape (n_samples,)
            Density-based weights.
        """
        # Compute density-based weights
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        elif isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy() # type: ignore

        kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(y_true.reshape(-1, 1))
        n = len(y_true)
        average_density = np.sum(np.subtract(1, kde.score_samples(y_true.reshape(-1, 1)))) / n
        density_weights = np.subtract(1, kde.score_samples(y_true.reshape(-1, 1))) / average_density

        DensityBasedWeighting.kde = kde
        DensityBasedWeighting.average_density = average_density
        DensityBasedWeighting.n = n
        DensityBasedWeighting.density_weights = density_weights

        return average_density

    @staticmethod
    @validate_params(
        {
            "y_true": ["array-like"],
            "y_pred": ["array-like"],
            "sample_weight": ["array-like", None],
            "multioutput": [StrOptions({"raw_values", "uniform_average"}), "array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def density_based_weighting(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
        """Density-based weighting for imbalanced datasets.
    
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth target values.
        y_pred : array-like of shape (n_samples,)
            Estimated target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        multioutput : string in ["raw_values", "uniform_average"], default="uniform_average"
            Defines aggregating of multiple output scores.
            Array-like value defines weights used to average scores.
    
        Returns
        -------
        score : float
            Density-based weighted score.
    
        References
        ----------
        .. [1] Steininger, M, et al. (2021).
               Density-based weighting for imbalanced regression. Springer Journal of Machine Learning,
               https://doi-org.proxy.bib.uottawa.ca/10.1007/s10994-021-06023-5
        """
        # Check inputs
        if multioutput not in ("raw_values", "uniform_average"):
            raise ValueError("Invalid 'multioutput' parameter: {}".format(multioutput))
    
        # Compute density-based weights
        y_true = y_true.to_numpy()

        if DensityBasedWeighting.density_weights is None:
            DensityBasedWeighting.compute_density_weights(y_true)

        density_weights = DensityBasedWeighting.density_weights

        errors = np.abs(np.subtract(y_true, y_pred))
        weighted_errors = np.multiply(errors, density_weights)
        score = np.mean(weighted_errors)
    
        return score





