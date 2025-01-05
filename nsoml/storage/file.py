from pathlib import Path
from typing import Optional
from pandas import DataFrame
import pandas as pd
import os
import pickle
from sklearn.base import BaseEstimator as Model
from .storage import BaseStorage

class FileStorage(BaseStorage):
    def __init__(self, 
                 base_path: Path, 
                 cv_name: str,
                 resampler: str, 
                 model: str, 
                 scorer: str, 
                 fold: str,
                 tag: Optional[str] = None,
                 metadata: Optional[dict] = None,
                 timestamp: Optional[str] = None):
        super().__init__(base_path, cv_name, resampler, model, scorer, fold, tag, metadata, timestamp)
        Warning('FileStorage is not yet implemented')            
            
        if not self.path.exists():
            os.makedirs(self.path, exist_ok=True)

    def save_result(self, df: DataFrame) -> None:
        out = self._results_file()
        df.to_csv(out, index=False)
        if self.metadata is not None:
            with open(self._metadata_file(), 'wb') as f:
                pickle.dump(self.metadata, f)

    def load_result(self) -> DataFrame:
        file = self._results_file()
        if not file.exists():
            raise FileNotFoundError(f'The file {file} does not exist')
        return pd.read_csv(file)
        
    def save_model(self, model: Model) -> None:

        with open(self._model_file(), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self) -> Model:
        with open(self._model_file(), 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, Model):
            raise TypeError('The loaded file is not of the type Model')
        return model
    
    def _results_file(self) -> Path:
        return self.path / f'results_{self.fold}.csv'
    
    def _metadata_file(self) -> Path:
        return self.path / f'metadata_{self.fold}.pkl'
    
    def _model_file(self) -> Path:
        return self.path / f'model_{self.fold}.pkl'
