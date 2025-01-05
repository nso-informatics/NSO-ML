from typing import Self, Optional, Dict, Callable, Tuple
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import os
from .cah import load_cah_data


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

class Preprocessor:
    def __init__(self,
                 analyte_dir: Path,
                 outcome_dir: Optional[Path],
                 storage_dir: Path,
                 outcomes: Optional[str] = None, 
                 file_format: str = "csv",
                 target_label: str = "definitive_diagnosis",
                 force: bool = False,):
        """
        Initializes the preprocessor with the given dataset.
        """
        self.data: pd.DataFrame = pd.DataFrame()
        self.analyte_dir = analyte_dir
        self.outcome_dir = outcome_dir
        self.storage_dir = storage_dir
        self.target_label = target_label
        self.file_format = file_format
        self.outcomes = outcomes
        self.force = force

        if not analyte_dir.exists():
            raise FileNotFoundError(f"Analyte directory {analyte_dir} does not exist.")
        if outcome_dir and not outcome_dir.exists():
            raise FileNotFoundError(f"Outcome directory {outcome_dir} does not exist.")
        if not storage_dir.exists():
            raise FileNotFoundError(f"Storage directory {storage_dir} does not exist.")

        if Path(storage_dir / f'analytes.{file_format}').exists() and not force:
            if file_format == "parquet":
                self.data = pd.read_parquet(storage_dir / f'analytes.{file_format}')
            elif file_format == "csv":
                self.data = pd.read_csv(storage_dir / f'analytes.{file_format}')
            else:
                raise ValueError(f"Unknown file format {file_format}.")
        else:
            self.auto_preprocess()


    def save(self):
        assert self.data is not None, "No data to save."
        if self.file_format == "parquet":
            self.data.to_parquet(self.storage_dir / f'analytes.{self.file_format}')
        elif self.file_format == "csv":
            self.data.to_csv(self.storage_dir / f'analytes.{self.file_format}')
        else:
            raise ValueError(f"Unknown file format {self.file_format}.")

    def auto_preprocess(self):
        """
        Automatically preprocesses the dataset.

        This includes:
        - Loading the analyte Data
        - Loading the outcomes Data
        - Removing positive screens
        - Renaming columns
        - Cleaning Values
        - Handling missing Values
        - Making calculations 
        - One hot encoding
        - Integer encoding 
        - Saving the data
        """
        print(self.storage_dir / "analytes_unprocessed.csv")
        if not (self.storage_dir / "analytes_unprocessed.csv").exists():
            self.load_analyte_data()
            print("Raw Unprocessed Analyte Data: ", self.data.shape)
        else:
            self.data = pd.read_csv(self.storage_dir / "analytes_unprocessed.csv", low_memory=False)
            print("Loaded Unprocessed Analyte Data: ", self.data.shape)
            print("Analyte Data: ", self.data)
            print("Columns: ", list(self.data.columns))

        if self.outcome_dir:
            self.load_outcomes_data()
            # self.remove_positive_screens()
            print("No Screen Positive Analyte Data: ", self.data.shape)

        print("Analyte Data: ", self.data)
        print("Columns: ", self.data.columns)

        self.name_mapping()
        self.clean_values()
        print("Cleaned Analyte Data: ", self.data.shape)
        assert 'definitive_diagnosis' in self.data.columns, self.data.columns

        self.handle_missing_values()
        print("Missing Values Removed: ", self.data.shape)

        self.make_calculations()
        print("Calculated Analyte Data: ", self.data.shape)

        self.one_hot_encoding()
        self.data = self.data.set_index("episode")
        print("Encoded Analyte Data: ", self.data.shape)

        self.save()

    def get_data(self, Xy: bool = False) -> pd.DataFrame | Tuple[pd.DataFrame, pd.Series]: 
        if Xy:
            self.target = self.data[self.target_label]
            possible_targets=['definitive_diagnosis', 'final_diagnosis', 'screen_result', 'initial_result']
            for target in possible_targets:
                if target in self.data.columns:
                    self.data.drop(columns=[target], inplace=True)
            return self.data, self.target
        else:
            return self.data

    def load_analyte_data(self, analyte_dir: Optional[Path] = None) -> Self:
        analyte_dir = analyte_dir if analyte_dir else self.analyte_dir
        analyte_regex = r'.*\d{8,8}-\d{8,8}_OMNINBSInitialRAWAnalytesAndScrDets_.*\.xlsx'
        dir_files = [file.name for file in analyte_dir.glob(r"*")]
        analyte_files = [file for file in dir_files if re.match(analyte_regex, str(file))]

        analyte_data = pd.DataFrame()
        for file in tqdm(analyte_files, desc="Loading analyte data"):
            file = analyte_dir / file
            analyte = pd.read_excel(file)
            analyte.rename(columns={"Accession Number": "Episode"}, inplace=True)
            analyte_data = pd.concat([analyte_data, analyte], ignore_index=True)

        # Save analyte data to disk
        analyte_data.to_csv(self.storage_dir / "analytes_unprocessed.csv", index=False)

        self.data = analyte_data
        return self

    def load_outcomes_data(self, outcomes_regex: Optional[str] = None) -> Self:
        if not self.outcome_dir:
            raise ValueError("Outcomes directory not provided.")
        if str(self.outcomes).upper() == "CH":
            regex = outcomes_regex if outcomes_regex else r'.*\d{8,8}-\d{8,8}_CHDiagnosticOutcomes_.*\.xlsx'
            dir_files = [file.name for file in self.outcome_dir.glob(r"*")]
            outcome_files = [file for file in dir_files if re.match(regex, str(file))]
            print("Outcome Files: ", outcome_files)
            outcomes_data = pd.DataFrame()
            for file in outcome_files:
                print(file, end=" ")
                file = self.outcome_dir / file
                print(convert_bytes(os.path.getsize(file)))
                outcomes = pd.read_excel(file)
                outcomes.rename(columns={"Accession Number": "Episode"}, inplace=True)
                outcomes_data = pd.concat([outcomes_data, outcomes], ignore_index=True)
    
            self.data = pd.merge(self.data, outcomes_data, on="Episode", how="left")
        elif str(self.outcomes).upper() == "CAH":
            files = list(self.outcome_dir.glob("*.sqlite"))
            assert len(files) == 1, "Multiple sqlite3 files found."
            outcomes_data = load_cah_data(files[0])
            self.data = pd.merge(self.data, outcomes_data, how="left", left_on="Episode", right_on="sample_id")
            self.data.drop(columns=["sample_id"], inplace=True)
            print(self.data)
            print(list(self.data.columns))

        return self


    def name_mapping(self, rules: dict[str, str] = {}) -> Self:
        """
        Renames the columns of the dataset. Any names not defined in the mapping will be removed.
        """
        # Join the default mapping with the given mapping rules
        rules = {**default_name_mapping(), **rules}
        removed_columns = [x for x in self.data.columns if x not in rules.keys()]
        self.data = self.data.drop(columns=removed_columns)
        self.data = self.data.rename(columns=rules)
        return self

    def clean_values(self, custom_rules: Dict[str, Dict[re.Pattern, Callable]] = {}, replace_rules: bool = False) -> Self:
        """
        Clean values in the dataset.
        
        For example, we can convert strings to integers, floats, or booleans.
        We can also drop rows based on the values in the dataset.
        Any row with NaN values will be dropped.
        
        :param custom_rules: A dictionary of column names and their corresponding rules. This will override the default rules when conflicting.
        :param replace_rules: If True, the custom rules will replace the default rules.
        """
        # Regular expression for numeric values (integers, floats, scientific notation)
        numeric_regex = re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$")         
        default_behaviour: Dict[re.Pattern, Callable] = {
            numeric_regex: lambda x: abs(float(x)),
            re.compile(r"^(Not Tested)$"): lambda _: None,
        }

        
        rules: Dict[str, Dict[re.Pattern, Callable]] = {
            "episode": {
                re.compile(r".*"): lambda x: x,
            },
            "gestational_age": {
                numeric_regex: lambda x: abs(int(x)),
            },
            "birth_weight": {
                numeric_regex: lambda x: abs(int(x)),
            },           
            "sex": {
                re.compile(r".*(M|F).*"): lambda x: 0 if 'fe' in str(x).lower() else 1,
                re.compile(r".*(Ambiguous).*"): lambda _: 2, # Ambiguous is mapped into M=1 and F=1
                re.compile(r".*(Unknown).*"): lambda _: 3, # Unknown is mapped into M=0 and F=0
            },
            "age_at_collection": {
                numeric_regex: lambda x: abs(float(x)) if abs(float(x)) > 24.0 and abs(float(x)) < 168.0 else None, # Drop rows with age outside of 1-7 days 
            },
            "transfusion_status": {
                re.compile(r"N"): lambda _: 0,
                re.compile(r"Y"): lambda _: 1,
                re.compile(r"U"): lambda _: 0,
            },
            "multiple_birth_rank": {
                re.compile(r"^(nan)$"): lambda _: 0,
                # There are issues with categorical values and sweetvix. 
                # Best to leave as boolean for now. Also do not know the impact
                # of multiple births on the outcome. (bool vs cat.)
    #            re.compile(r"^[a-zA-Z]$"): lambda x: ord(str(x).lower()) - 97 if str(x).isalpha() else None
                re.compile(r"^[a-zA-Z]$"): lambda _: 1
            },
            "HGB_Pattern": {
                re.compile(r"^FA$"): lambda _: 1,
                re.compile(r".*"): lambda _: 0,
            },
            "definitive_diagnosis": {
                re.compile(r"^(nan)$"): lambda _: 0,
                re.compile(r"Negative"): lambda _: 0,
                re.compile(r"Positive"): lambda _: 1,
                re.compile(r"Unknown"): lambda _: None,
            },
            'screen_result': {
                re.compile(r"^(nan)$"): lambda _: 0,
                re.compile(r"Negative"): lambda _: 0,
                re.compile(r"POSITIVE"): lambda _: 1,
                re.compile(r"UNSATISFACTORY"): lambda _: None,
            },
            'initial_result': {
                re.compile(r"Negative"): lambda _: 0,
                re.compile(r"Request confirm"): lambda _: 1,
                re.compile(r"UNSATISFACTORY"): lambda _: None,
                re.compile(r"Remove"): lambda _: None,
                re.compile(r".*"): lambda _: 0,
            },
        }

        if replace_rules:
            rules = custom_rules
        else:
            rules.update(custom_rules)
            
        # Set all columns to default rules and update with specific rules
        mapping: Dict[str, Dict] = {}
        for column in self.data.columns:
            mapping[column] = default_behaviour
        mapping.update(rules)

        print(list(self.data.columns))
        print(self.data['screen_result'].value_counts())
       
        # Apply the first rule that matches the value in the column
        def apply_rule(x, rule):
            for key, value in rule.items():
                if key.match(str(x)):
                    return value(x)
            # If no rule matches, we drop the row as it is erroneous
            return None
        
        for column in tqdm(mapping, desc="Cleaning values"):
            self.data[column] = self.data[column].apply(lambda x: apply_rule(x, mapping[column]))
    
        print(self.data)
        print(self.data['episode'])
        return self

    def handle_missing_values(self) -> Self:
        self.data.dropna(inplace=True)
        return self

    def remove_positive_screens(self) -> Self:
        """
        Remove rows with positive screens.
        """
        for column in self.data.columns:
            if "ScrDet" in column and not self.outcomes in column:
                if "MPS1" in column:
                    self.data = self.data.drop(columns=[column])
                    continue

                print(f"Removing positive screens in {column}.")
                # Case to string to avoid errors
                self.data[column] = self.data[column].astype(str)
                len_not_negative = len(self.data[~self.data[column].str.contains("Negative") & ~self.data[column].str.contains("Not Detected")])
                self.data = self.data[self.data[column].str.contains("Negative") | self.data[column].str.contains("Not Detected")]
                print(f"Removed {len_not_negative} rows with positive screens in {column}.")
        assert len(self.data) > 0, "No rows left after removing positive screens."
        return self

    def make_calculations(self) -> Self:
        return self

    def one_hot_encoding(self) -> Self:
        """
        Perform one hot encoding on the dataset by mapping values into boolean columns.
        """
        columns: Dict[str, Dict] = {
            # This is an example of how to one hot encode a column
            # "column": {
            #    value: label,
            #    value: label
            #}
            "sex": {
                0: ["sex_female"],
                1: ["sex_male"],
                2: ["sex_male", "sex_female"],
            }
        }
        
        # One hot encode columns using the columns dictionary
        new_columns = []
        for column, mapping in columns.items():
            for value, labels in mapping.items():
                for label in labels:
                    new_columns.append(label)
        new_columns = list(set(new_columns))
    
        print(f"Adding {len(new_columns)} new columns. {new_columns}")
        print(new_columns)
        for column in new_columns:
            self.data[column] = 0
    
        for column, mapping in columns.items():
            for row in tqdm(self.data.iterrows(), desc=f"One hot encoding {column}", total=len(self.data)):
                for value, labels in mapping.items():
                    for label in labels:
                        if row[1][column] == value:
                            self.data.at[row[0], label] = 1                
    
            self.data.drop(columns=[column], inplace=True)
        return self

def default_name_mapping() -> dict[str, str]:
    """
    Returns a dictionary mapping the original column names to the new column names.

    :return: Dictionary mapping
    """
    return {
        # Original: New
        "Episode": "episode",
        "sample_id": "episode",
        "Definitive Diagnosis": "definitive_diagnosis",
        "Gestational Age (days)": "gestational_age",
        "Birth Weight (g)": "birth_weight",
        "Sex": "sex",
        "Age at Collection (hr)": "age_at_collection",
        "Transfusion Status": "transfusion_status",
        "Multiple Birth Rank": "multiple_birth_rank",
        "ALA_I (RAW)": "ALA",
        "ARG_I (RAW)": "ARG",
        "CIT_I (RAW)": "CIT",
        "GLY_I (RAW)": "GLY",
        "LEU_I (RAW)": "LEU",
        "MET_I (RAW)": "MET",
        "ORN_I (RAW)": "ORN",
        "PHE_I (RAW)": "PHE",
        "SUAC_I (RAW)": "SUAC",
        "TYR_I (RAW)": "TYR",
        "VAL_I (RAW)": "VAL",
        "C0_I (RAW)": "C0",
        "C2_I (RAW)": "C2",
        "C3_I (RAW)": "C3",
        "C3DC_I (RAW)": "C3DC",
        "C4_I (RAW)": "C4",
        "C4DC_I (RAW)": "C4DC",
        "C4OH_I (RAW)": "C4OH",
        "C5_I (RAW)": "C5",
        "C5:1_I (RAW)": "C5:1",
        "C5OH_I (RAW)": "C5OH",
        "C5DC_I (RAW)": "C5DC",
        "C6_I (RAW)": "C6",
        "C6DC_I (RAW)": "C6DC",
        "C8_I (RAW)": "C8",
        "C8:1_I (RAW)": "C8:1",
        "C10_I (RAW)": "C10",
        "C10:1_I (RAW)": "C10:1",
        "C12_I (RAW)": "C12",
        "C12:1_I (RAW)": "C12:1",
        "C14_I (RAW)": "C14",
        "C14:1_I (RAW)": "C14:1",
        "C14:2_I (RAW)": "C14:2",
        "C14OH_I (RAW)": "C14OH",
        "C16_I (RAW)": "C16",
        "C16OH_I (RAW)": "C16OH",
        "C16:1OH_I (RAW)": "C16:1OH",
        "C18_I (RAW)": "C18",
        "C18:1_I (RAW)": "C18:1",
        "C18:2_I (RAW)": "C18:2",
        "C18OH_I (RAW)": "C18OH",
        "C18:1OH_I (RAW)": "C18:1OH",
        "BIOT_I (RAW)": "BIOT",
        "GALT_I (RAW)": "GALT",
        "IRT_I (RAW)": "IRT",
        "TREC QN_I (RAW/CALC)": "TREC_QN",
        "TSH_I (RAW)": "TSH",
        "HGB Pattern_I (RAW)": "HGB_Pattern",
        "A_I (RAW)": "A",
        "F_I (RAW)": "F",
        "F1_I (RAW)": "F1",
        "FAST_I (RAW)": "FAST",
        'sample_id': 'episode',
        'ga': 'gestational_age',
        'bw': 'birth_weight',
        'final_diagnosis': 'definitive_diagnosis',
        'definitive_diagnosis': 'definitive_diagnosis',
        'screen_result': 'screen_result',
        'initial_result': 'initial_result',
        'N17OHP_I (RAW)': '17OHP',
    }

