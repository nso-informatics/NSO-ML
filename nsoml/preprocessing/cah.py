import pandas as pd
import sqlite3
from pathlib import Path


def load_cah_data(
    sqlite_path: Path = Path("/data/CAHML/cah_star.sqlite"),
    steroids: bool = False,
):

    conn = sqlite3.connect(sqlite_path)
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(tables_query, conn)
    dataframes = {}
    for table in tables["name"].to_list():
        query = f"SELECT * FROM {table}"
        df = pd.read_sql_query(query, conn)
        dataframes[table] = df
    conn.close()

    df = dataframes["sample_dimension"]
    tables_to_merge = [
        "demographics_dimension",
        "determination_dimension",
        "determination_dimension_temp",
        "diagonsis_dimension",
        "lab_test_result_fact",
    ]

    for table in tables_to_merge:
        if table in dataframes:
            df = pd.merge(
                df,
                dataframes[table],
                on="sample_id",
                how="left",
                suffixes=("", f"_{table}"),
            )

    columns_to_remove = [
        "birth_date",
        "birth_time",
        "collection_date",
        "received_date",
        "received_time",
        "test_date",
        "test_time",
        "date_key",
        "date",
        "year_month",
        "year",
        "month",
        "day",
        "day_number_in_year_transfused_date",
        "date_key_transfused_date",
        "date_transfused_date",
        "year_month_transfused_date",
        "year_transfused_date",
        "month_transfused_date",
        "day_transfused_date",
        "time_key",
        "time",
        "hour",
        "minute",
        "time_key_test_time",
        "time_test_time",
        "hour_test_time",
        "minute_test_time",
        "test_key",
        "case_num",
        "day_number_in_year",
        "final_determination_determination_dimension",
        "initial_determination_determination_dimension",
        "determination_type_determination_dimension",
        "diagnostic_determination_determination_dimension",
    ]
    df.drop(columns=columns_to_remove, inplace=True, errors="ignore")
    if steroids:
        df = df.pivot(index="sample_id", columns="test_type", values="test_result")
    else:
        df = df.drop(columns=["test_type", "test_result"], inplace=True)

    return df


def join_analytes(analytes: pd.DataFrame, cah: pd.DataFrame):
    """
    """
    cah_key = "sample_id" if "sample_id" in cah.columns else "episode"
    analyte_key = 'episode'

    cah = cah.set_index(cah_key)
    analytes = analytes.set_index(analyte_key)

    df = cah.join(analytes, how="left")
    assert len(df) == len(cah)
    assert len(df) == len(analytes)
    return df
