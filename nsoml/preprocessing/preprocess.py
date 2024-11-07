#from utils.export import export
from typing import Self, Optional, Dict, Callable
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import os

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
                 data: pd.DataFrame = pd.DataFrame(),
                 origin_dir: Path = Path('/data/CAHML/GA/data/initials_and_outcomes'),
                 outcomes_dir: Optional[Path] = None,
                 file_format: str = "csv",
                 force: bool = False,):
        """
        Initializes the preprocessor with the given dataset.
        """
        self.data = data
        self.origin_dir = origin_dir
        self.outcomes_dir = outcomes_dir
        self.file_format = file_format

        if Path('/data/CAHML/GA/data/analytes.parquet').exists() and not force and file_format == "parquet":
            self.data = pd.read_parquet('/data/CAHML/GA/data/analytes.parquet')
        elif Path('/data/CAHML/GA/data/analytes.csv').exists() and not force and file_format == "csv":
            self.data = pd.read_csv('/data/CAHML/GA/data/analytes.csv')
        else:
            self.auto_preprocess()

    def save(self, file_format: str = "csv"):
        if file_format == "parquet":
            self.data.to_parquet('/data/CAHML/GA/data/analytes.parquet')
        elif file_format == "csv":
            self.data.to_csv('/data/CAHML/GA/data/analytes.csv')

    def auto_preprocess(self):
        self.load_analyte_data()
        if self.outcomes_dir:
            self.load_outcomes_data()
        print("Raw Unprocessed Analyte Data: ", self.data.shape)
        self.remove_positive_screens()
        self.name_mapping()
        print("No Screen Positive Analyte Data: ", self.data.shape)
        self.clean_values()
        print("Cleaned Analyte Data: ", self.data.shape)
        self.handle_missing_values()
        print("Missing Values Removed: ", self.data.shape)
        self.make_calculations()
        self.one_hot_encoding()
        self.integer_encoding()
        self.data = self.data.set_index("episode")
        self.save('csv')
        self.save('parquet')

    def load_analyte_data(self, analyte_dir: Optional[Path] = None) -> Self:
        analyte_dir = analyte_dir if analyte_dir else self.origin_dir
        analyte_regex = r'.*\d{8,8}-\d{8,8}_OMNINBSInitialRAWAnalytesAndScrDets_.*\.xlsx'
        dir_files = [file.name for file in analyte_dir.glob(r"*")]
        analyte_files = [file for file in dir_files if re.match(analyte_regex, str(file))]

        analyte_data = pd.DataFrame()
        for file in tqdm(analyte_files, desc="Loading analyte data"):
            file = analyte_dir / file
            analyte = pd.read_excel(file)
            analyte.rename(columns={"Accession Number": "Episode"}, inplace=True)
            analyte_data = pd.concat([analyte_data, analyte], ignore_index=True)

        self.data = analyte_data
        return self

    def load_outcomes_data(self, outcomes_regex: Optional[str] = None, outcomes_dir: Optional[Path] = None) -> Self:
        outcomes_dir = outcomes_dir if outcomes_dir else self.outcomes_dir
        if not outcomes_dir:
            raise ValueError("Outcomes directory not provided.")
        regex = outcomes_regex if outcomes_regex else r'.*\d{8,8}-\d{8,8}_CHDiagnosticOutcomes_.*\.xlsx'
        dir_files = [file.name for file in outcomes_dir.glob(r"*")]
        outcome_files = [file for file in dir_files if re.match(regex, str(file))]
        print("Outcome Files: ", outcome_files)
        outcomes_data = pd.DataFrame()
        for file in outcome_files:
            print(file, end=" ")
            file = outcomes_dir / file
            print(convert_bytes(os.path.getsize(file)))
            outcomes = pd.read_excel(file)
            outcomes.rename(columns={"Accession Number": "Episode"}, inplace=True)
            outcomes_data = pd.concat([outcomes_data, outcomes], ignore_index=True)

        self.data = pd.merge(self.data, outcomes_data, on="Episode", how="left")
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
       
        # Apply the first rule that matches the value in the column
        def apply_rule(x, rule):
            for key, value in rule.items():
                if key.match(str(x)):
                    return value(x)
            # If no rule matches, we drop the row as it is erroneous
            return None
        
        for column in tqdm(mapping, desc="Cleaning values"):
            self.data[column] = self.data[column].apply(lambda x: apply_rule(x, mapping[column]))
    
        return self

    def handle_missing_values(self) -> Self:
        self.data.dropna(inplace=True)
        return self

    def remove_positive_screens(self) -> Self:
        """
        Remove rows with positive screens.
        """
        for column in self.data.columns:
            if "ScrDet" in column:
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
                0: ["sex_male"],
                1: ["sex_female"],
                2: ["sex_male", "sex_female"],
            }
        }
        
        # One hot encode columns using the columns dictionary
        new_columns = []
        for column, mapping in columns.items():
            for value, labels in mapping.items():
                for label in labels:
                    new_columns.append(label)
    
        self.data[new_columns] = np.zeros((len(self.data), len(new_columns)))
    
        for column, mapping in columns.items():
            for row in tqdm(self.data.iterrows(), desc=f"One hot encoding {column}", total=len(self.data)):
                for value, labels in mapping.items():
                    for label in labels:
                        if row[1][column] == value:
                            self.data.at[row[0], label] = 1                
    
            self.data.drop(columns=[column], inplace=True)
        return self

    def integer_encoding(self) -> Self:
        for column in self.data.columns:
            if self.data[column].dtype == float:
                if np.all(self.data[column] == self.data[column].astype(int)):
                    self.data[column] = self.data[column].astype(int)
                
        return self

def default_name_mapping() -> dict[str, str]:
    """
    Returns a dictionary mapping the original column names to the new column names.

    :return: Dictionary mapping
    """
    return {
        # Original: New
        "Episode": "episode",
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
    }
