
import pandas as pd
from numpy.typing import ArrayLike

import warnings
warnings.filterwarnings("ignore")


def SGA(df: pd.DataFrame, p: int | float) -> ArrayLike:
    """
    Calculate which babies are Small for Gestational Age (SGA) based on the given percentile of birth weight
    The babies are binned based on gestational age in weeks (27 weeks, 28 weeks, etc...)
    Results are seperated by sex.

    :param df: DataFrame with columns 'gestational_age', 'sex_male', 'sex_female', 'birth_weight'
    :param p: The percentile to use for the SGA calculation
    :return: The indices that are underweight given percentile
    """
    # Bin the birth weight based on gestational age (weeks)
    bins = pd.cut(df['gestational_age'] / 7.0, bins=range(26, 45), right=False)
    df['bin'] = bins

    # Calculate the percentile of birth weight for each bin
    df.loc[:, 'percentile'] = df.groupby(['bin', 'sex_male', 'sex_female'])['birth_weight'].transform(lambda x: x.quantile(p))

#    print(df['percentile'].describe())
#    print(df['percentile'].head())

    # Determine which babies are SGA
    sga = df['birth_weight'] < df['percentile']
#    print(sga.count())
#    print(sga.sum())
#    print(sga.describe())
    return sga

