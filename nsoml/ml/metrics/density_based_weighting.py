from typing import List, Union
from sklearn.utils._param_validation import validate_params, StrOptions
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
    indices = None
    density_samples = None
    density_samples_input = None

    @staticmethod
    def compute_density_weights(y_true: Union[List, np.ndarray]) -> None:
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
            y_true = y_true.to_numpy()  # type: ignore

        kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(y_true.reshape(-1, 1))
        n = len(y_true)
        #indices = np.random.randint(0, n, 10000)
        #density_samples = kde.score_samples(y_true.reshape(-1, 1)[indices])  # log density
        indices = np.random.uniform(min(y_true), max(y_true), 500)
        density_samples = kde.score_samples(indices.reshape(-1, 1))  # log density
        density_samples = np.exp(density_samples)  # density

        DensityBasedWeighting.kde = kde
        DensityBasedWeighting.density_samples = density_samples
        DensityBasedWeighting.indices = indices
        #DensityBasedWeighting.density_samples_input = y_true.reshape(-1, 1)[indices]
        DensityBasedWeighting.density_samples_input = indices
        DensityBasedWeighting.n = n

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

        assert DensityBasedWeighting.density_samples is not None
        assert DensityBasedWeighting.density_samples_input is not None

        density_samples = np.interp(
            x=y_true,
            xp=DensityBasedWeighting.density_samples_input.reshape(-1), # Reference inputs
            fp=DensityBasedWeighting.density_samples , # Reference densities
        ) # This is our density function now.

        average_density = np.sum(density_samples) / len(density_samples)


        #average_density = np.sum(np.subtract(1, density_samples)) / len(density_samples)
        #density_weights = np.subtract(1, density_samples) / average_density
        errors = np.abs(np.subtract(y_true, y_pred))
        weighted_errors = np.multiply(errors, density_weights)
        score = np.mean(weighted_errors)
        print("Density-based weighted score: {:.4f}".format(score))

        DensityBasedWeighting.density_weights = density_weights
        DensityBasedWeighting.average_density = average_density


        return score
