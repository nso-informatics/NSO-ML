import numpy as np
import pandas as pd
import math
import random as rand

class GaussianResampler():

    def __init__(self, feature_ratio: float, noise_level: float, new_points_ratio: float, replacement: bool = True):
        """
        Resample the minority class using Gaussian noise.
        
        Featue selection is performed randomly on a point by point basis.
        The noise level is a multiplier for the standard deviation of the Gaussian noise.
        The new_points_ratio is the ratio of new points to generate for each existing minority point.
        Opting to use replacement will add the resampled points to the original data, otherwise the resampled data will replace the original data.

        :param data: The input data, a 2D numpy array of shape (num_samples, num_features)
        :param feature_ratio: The ratio of features to keep
        :param noise_level: The standard deviation of the Gaussian noise to add
        :param new_points: The ratio of new points to existing minority points
        :return: The resampled data
        """
        assert 0 < feature_ratio <= 1, "feature_ratio must be between 0 and 1"
        assert noise_level >= 0, "noise_level must be non-negative"
        assert new_points_ratio >= 0, "new_points must be non-negative" 
        
        self.feature_ratio = feature_ratio
        self.noise_level = noise_level
        self.new_points_ratio = new_points_ratio
        self.replacement = replacement

    def __repr__(self):
        return f"GaussianResampler"

    def __str__(self):
        return f"GaussianResampler"
        
    def __call__(self, *args, **kwds):
        return self.fit_resample(*args, **kwds)
        
    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), "X and y must have the same length"
        assert len(X) > 0, "X must have at least one sample"
        assert len(X) > 0, "y must have at least one sample"
        assert len(X.shape) == 2, "X must be a 2D array"

        columns = None
        if isinstance(X, pd.DataFrame):
            columns = X.columns
            X = X.values
        
        _, num_columns = X.shape
        resampled = []
        float_column_indices = []
        for col_index in range(num_columns):
            try:
                # This will raise a ValueError if the column is not numeric
                float_col = X[:, col_index].astype(float) 
                float_column_indices.append(col_index)
            except ValueError:
                continue
                
        # Choose the minority class
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        minority_X = X[y == minority_class]
        majority_X = X[y != minority_class]
        minority_Y = y[y == minority_class]
        majority_Y = y[y != minority_class]
        
        # Build column information for resampling
        column_dist = []
        for col_index in float_column_indices:
            float_col = minority_X[:, col_index].astype(float)
            mean = np.mean(float_col)
            std_dev = np.std(float_col)
            column_dist.append((col_index, mean, std_dev))

        # Determine the number of new points to generate for each existing minority point
        total_new_point_count = math.ceil(len(minority_X) * self.new_points_ratio)
        resample_count_by_row = np.array([0] * len(minority_X))
        while np.sum(resample_count_by_row) < total_new_point_count:
            resample_count_by_row[rand.randint(0, len(minority_X) - 1)] += 1
        
        # Generate the new points
        for index, row in enumerate(minority_X):
            for _ in range(resample_count_by_row[index]):
                new_row = row.copy()
                resampled_column_indices = np.random.choice(float_column_indices, int(self.feature_ratio * len(float_column_indices)), replace=False)
                for col_index, mean, std_dev in column_dist:
                    if col_index in resampled_column_indices:
                        new_row[col_index] += np.random.normal(0, std_dev * self.noise_level)
                resampled.append(new_row)
                
        # Ensure that the resampled data is not empty
        if len(resampled) == 0:
            return (X, y)        
        
        # Join minority and majority classes
        if self.replacement:
            out = (
                np.concatenate((minority_X, majority_X, resampled), axis=0),
                np.concatenate((minority_Y, majority_Y, np.full((len(resampled),), minority_class)), axis=0)    
            )
        else:
            out = (
                np.concatenate((majority_X, resampled), axis=0),
                np.concatenate((majority_Y, np.full((len(resampled),), minority_class)), axis=0)    
            )
        if columns is not None:
            out = (pd.DataFrame(out[0], columns=columns), pd.Series(out[1]))
        return out
