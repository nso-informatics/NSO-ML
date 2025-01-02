import sqlite3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    


def load_cah_data(file_path):
    """
    Load the data from the file and return a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the file.

    Returns
    -------
    pd.DataFrame
        The data in a pandas DataFrame.
    """
    
        # Connect to the SQLite database and fetch table names
    conn = sqlite3.connect(str(file_path))
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
    
    # Load all tables into a dictionary of DataFrames
    dataframes = {table: pd.read_sql(f"SELECT * FROM {table};", conn) for table in tables}
    conn.close()
    
    # Merge selected tables on 'sample_id'
    merged_df = dataframes['sample_dimension']
    tables_to_merge = [
        'demographics_dimension',
        'determination_dimension_temp',
        'determination_dimension',
        'diagnosis_dimension',
        'lab_test_result_facts'
    ]
    for table in tables_to_merge:
        if table in dataframes:
            merged_df = pd.merge(merged_df, dataframes[table], on='sample_id', how='left', suffixes=('', f'_{table}'))
    
    # Handle duplicate columns and reset index
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df.reset_index(drop=True, inplace=True)
    
    # Remove irrelevant columns
    columns_to_remove = [
        'birth_date', 'birth_time', 'collection_date', 'received_date', 'received_time', 'test_date', 'test_time',
        'date_key', 'date', 'year_month', 'year', 'month', 'day', 'day_number_in_year_transfused_date',
        'date_key_transfused_date', 'date_transfused_date', 'year_month_transfused_date', 'year_transfused_date',
        'month_transfused_date', 'day_transfused_date', 'time_key', 'time', 'hour', 'minute', 'time_key_test_time',
        'time_test_time', 'hour_test_time', 'minute_test_time', 'test_key', 'case_num', 'day_number_in_year',
        'final_determination_determination_dimension', 'initial_determination_determination_dimension',
        'determination_type_determination_dimension', 'diagnostic_determination_determination_dimension'
    ]
    merged_df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    
    # Pivot the DataFrame and handle 'test_result'
    results_df = merged_df.pivot(index='sample_id', columns='test_type', values='test_result')
    results_df.reset_index(inplace=True)
    results_df.set_index('sample_id', inplace=True)
    merged_df = merged_df.drop(columns='test_result', errors='ignore')
    
    # Map final diagnosis values
    final_diagnosis_mapping = {
        np.nan: 'Negative', None: 'Negative', pd.NaT: 'Negative', pd.NA: 'Negative',
        'Negative': 'Negative',
        'Interprovincial Emigration': 'Unknown',
        '21-Hydroxylase Deficiency: Salt Wasting': 'Positive',
        '21-Hydroxylase Deficiency: Simple Virilizing': 'Positive',
        '21-Hydroxylase Deficiency: Non-Classical': 'Positive',
        '21-Hydroxylase Deficiency: Classical': 'Positive',
        'Other': 'Unknown',
        '"Other': 'Unknown',
        'Positive': 'Positive',
        'Deceased - No Diagnosis': 'Unknown',
        'Deceased - Sepsis': 'Unknown',
        'Deceased - Other': 'Unknown',
        'Deceased - Complications of Prematurity': 'Unknown',
        'Deceased - Pulmonary Hemorrhage': 'Unknown',
        'Carrier': 'Unknown',
        '3-Beta Dehydrogenase Deficiency': 'Unknown',
        '11-Beta Hydroxylase Deficiency': 'Unknown',
        'Not Affected': 'Negative',
        'Investigations Declined by Parents': 'Unknown',
        'Persistant lab abnormalities': 'Unknown',
    }
    merged_df['final_diagnosis'] = merged_df['final_diagnosis'].map(final_diagnosis_mapping)
    merged_df = merged_df[merged_df['final_diagnosis'].notnull()]
    merged_df = merged_df[merged_df['final_diagnosis'] != 'Unknown']
    results_df = results_df[results_df['ia17ohp'].notnull()]
    merged_df['ia17ohp'] = results_df['ia17ohp']

    # Remove all sample_id that are integers
    def handle_episode(x):
        x.apply(lambda y: re.sub(r"^\D|-|N", "", str(y)))
        # x.apply(lambda y: f'N{str(y)[:8]}-{str(y)[8:]}')
        return x

    merged_df['sample_id'] = handle_episode(merged_df['sample_id'])
    


    # Drop irrelevant columns and rename
    df = merged_df.drop(columns=[
        'test_type', 'virilized', 'treatment', 'transfused_status', 'transfused_date',
       'screening_classification', 'prior_diagnosis', 'symptomatic', 'determination_type',
        'diagnostic_determination', 'screening_determination',  'ga', 'bw', #'initial_determination',
    ]).set_index('sample_id').drop_duplicates()
    
    df.rename(columns={
        'final_determination': 'screen_result',
        'final_diagnosis': 'definitive_diagnosis',
        'initial_determination': 'initial_result',
    }, inplace=True)

    # Add sample_id back to the df
    df['sample_id'] = df.index
    df.reset_index(drop=True, inplace=True)

    return df

    # Convert 'sample_id' to Integer
    df['sample_id'] = df['sample_id'].apply(lambda x: int(re.sub(r"[^0-9]", "", str(x))))
    df['sample_id'] = df['sample_id'].astype(int)


    print(df)
    print(df[df['sample_id'].astype(int) > 201907290001]['definitive_diagnosis'].value_counts())
    print(df[df['sample_id'].astype(int) > 201907290001]['screen_result'].value_counts())
    print(df[df['sample_id'].astype(int) > 201907290001]['initial_result'].value_counts())
    
    # Plot the average GA over time
    # plt.figure(figsize=(12, 6))
    # sns.lineplot(data=merged_df, x='sample_id', y='ga')
    # plt.title('Average GA Over Time')
    # plt.xlabel('sample_id')
    # plt.ylabel('ga')
    # plt.savefig('ga_over_time.png')
    
        
    return df
