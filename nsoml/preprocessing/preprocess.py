from utils.export import export
from typing import Self, Optional

@export
def test_preprocessing():
    print("Preprocessing data...")
    # do some preprocessing
    print("Data preprocessed!")



class Preprocessor:
    def __init__(self, data):
        """
        Initializes the preprocessor with the given dataset.
        """
        self.data = data

    def preprocess(self):
        self.name_mapping()
        self.clean_values()
        self.handle_missing_values()
        self.make_calculations()
        self.one_hot_encoding()
        self.integer_encoding()

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

    def clean_values(self) -> Self:
        return self

    def handle_missing_values(self) -> Self:
        return self

    def make_calculations(self) -> Self:
        return self

    def one_hot_encoding(self) -> Self:
        return self

    def integer_encoding(self) -> Self:
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
