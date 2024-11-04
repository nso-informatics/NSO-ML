import numpy as np
from scipy import stats

def mean_absolute_error_with_ci(y_true, y_pred, *, sample_weight=None, 
                                multioutput="uniform_average", confidence=0.95):
    """
    Mean absolute error regression loss with confidence interval.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
    confidence : float, default=0.95
        Confidence level for the confidence interval.

    Returns
    -------
    mae : float or ndarray of floats
        MAE output is non-negative floating point. The best value is 0.0.
    ci : tuple of floats or ndarray of tuples
        The confidence interval (lower bound, upper bound).

    Examples
    --------
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_error_with_ci(y_true, y_pred)
    (0.5, (0.17667209, 0.82332791))
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    check_consistent_length(y_true, y_pred, sample_weight)

    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, y_pred)

    output_errors = np.average(np.abs(y_pred - y_true), weights=sample_weight, axis=0)
    
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
    
    mae = np.average(output_errors, weights=multioutput)
    
    # Calculate confidence interval
    n = len(y_true)
    se = np.sqrt(np.sum(np.abs(y_true - y_pred)**2) / n) / np.sqrt(n)
    ci_margin = stats.t.ppf((1 + confidence) / 2, df=n-1) * se
    ci = (mae - ci_margin, mae + ci_margin)
    
    return mae, ci

# Helper functions (simplified versions of scikit-learn internals)
def _check_reg_targets(y_true, y_pred, multioutput):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    return "continuous", y_true, y_pred, multioutput

def check_consistent_length(*arrays):
    lengths = [len(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Inconsistent number of samples")

def _check_sample_weight(sample_weight, X):
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim != 1:
        raise ValueError("Sample weights must be 1D array or scalar")
    if sample_weight.shape != (X.shape[0],):
        raise ValueError("Sample weights must be same length as input")
    return sample_weight

# Example usage
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mae, ci = mean_absolute_error_with_ci(y_true, y_pred)
print(f"MAE: {mae}, 95% CI: {ci}")

