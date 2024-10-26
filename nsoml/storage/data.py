import pandas as pd
from utils.export import export
from typing import Optional

class DataSource:
    data: Optional[pd.DataFrame] = None

    def __init__(self, path: str):
        self.load_data(path)

    def load_data(self, path: str):
        self.data = pd.read_csv(path)

    def copy_data(self):
        if self.data is None:
            raise ValueError("Data is not loaded.")
        return self.data.copy()

@export
def load_data(path: str) -> pd.DataFrame:
    DataSource(path)
    if DataSource.data is None:
        raise ValueError("Data is not loaded.")
    return DataSource.data
    
