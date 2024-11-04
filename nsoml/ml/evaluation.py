
import pandas as pd
from numpy.typing import ArrayLike


def SGA(df: pd.DataFrame, p: int | float, weeks: bool) -> ArrayLike:
    """
    Calculate which babies are Small for Gestational Age (SGA) based on the given percentile of birth weight
    The babies are binned based on gestational age in weeks (27 weeks, 28 weeks, etc...)
    Results are seperated by sex.

    :param df: DataFrame with columns 'gestational_age', 'sex_male', 'sex_female', 'birth_weight'
    :param p: The percentile to use for the SGA calculation
    :param weeks: Whether the gestational age is in weeks or days
    :return: The indices that are underweight given percentile
    """

    if not weeks:
        df['gestational_age'] = df['gestational_age'] / 7

    # Calculate the percentile of the birth weight
    groups = df.groupby(['gestational_age', 'sex_male', 'sex_female'])['birth_weight']
    df['percentile'] = groups.transform(lambda x: x.quantile(p))

    # Find the indices of the babies that are underweight
    underweight = df['birth_weight'] < df['percentile']
    return underweight

