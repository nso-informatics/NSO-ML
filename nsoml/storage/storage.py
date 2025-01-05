from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator, KFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.base import BaseEstimator as Model

class BaseStorage(ABC):    
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
        self.base_path = base_path
        self.cv_name = cv_name
        self.resampler = resampler
        self.model = model
        self.scorer = scorer
        self.fold = fold
        self.tag = tag
        self.metadata = metadata
        self.timestamp = timestamp 
        
        self.path = base_path / cv_name / resampler / model / scorer / (tag if tag else '') / (timestamp if timestamp else '')
        
        super().__init__()

    @abstractmethod
    def save_result(self, df: DataFrame) -> None:
        pass

    @abstractmethod
    def load_result(self) -> DataFrame:
        pass

    @abstractmethod
    def save_model(self, model: Model) -> None:
        pass

    @abstractmethod
    def load_model(self) -> List[Model]:
        pass

    @classmethod
    def cross_validator_name(cls, cross_validator: BaseCrossValidator) -> str:
        if cross_validator is None:
            name = "none"
        elif type(cross_validator) == StratifiedKFold:
            name = f"StratifiedKFold_{cross_validator.n_splits}" # type: ignore
        elif type(cross_validator) == KFold:
            name = f"KFold_{cross_validator.n_splits}" # type: ignore
        elif type(cross_validator) == RepeatedStratifiedKFold:
            name = f"RepeatedStratifiedKFold_{cross_validator.n_repeats}x{cross_validator.cvargs['n_splits']}"
        else:
            name = cross_validator.__name__
        
        return name