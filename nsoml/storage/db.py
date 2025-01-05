import json
from pathlib import Path
from typing import List, Optional

from pandas import DataFrame
from .file import FileStorage
from .storage import BaseStorage
from .db_connection import get_connection
from ..analysis.analysis import Analysis, Filter
import psycopg2 as ps
import pandas as pd
import pickle

from sklearn.base import BaseEstimator as Model
from .storage import BaseStorage

class DBStorage(BaseStorage):
    conn = None

    def __init__(
        self,
        base_path: Path,
        cv_name: str,
        resampler: str,
        model: str,
        scorer: str,
        fold: str = "0",
        tag: Optional[str] = None,
        metadata: Optional[dict] = None,
        timestamp: Optional[str] = None,
    ):
        super().__init__(base_path, cv_name, resampler, model, scorer, fold, tag, metadata, timestamp)
        self._connect()
        if metadata is None:
            self.metadata = "{}"
        self.file_storage = FileStorage(base_path, cv_name, resampler, model, scorer, fold, tag, metadata, timestamp)

    def save_result(self, df: DataFrame):
        metric_db = df.copy()
        metric_db.insert(0, "cross_validator", self.cv_name)
        metric_db.insert(0, "score", self.scorer)
        metric_db.insert(0, "resampler", self.resampler)
        metric_db.insert(0, "model", self.model)
        metric_db.insert(0, "tag", self.tag)
        metric_db.insert(0, "fold", self.fold)

        analysis = Analysis.calculate_metrics(metric_db)
        analysis_json = json.dumps(analysis.loc[0].to_dict())
        self.file_storage.save_result(df)

        query = f"""
        BEGIN;
        
        DO $$
        DECLARE v_record_id INT;
        DECLARE v_combination_id INT;
        BEGIN
            SELECT id INTO v_combination_id 
            FROM combination 
            WHERE model = '{self.model}' 
                AND resampler = '{self.resampler}' 
                AND scorer = '{self.scorer}'
                AND cross_validator = '{self.cv_name}';
            IF NOT FOUND THEN
                INSERT INTO combination (model, resampler, scorer, cross_validator) 
                VALUES ('{self.model}', '{self.resampler}', '{self.scorer}', '{self.cv_name}')
                RETURNING id INTO v_combination_id;
            END IF;
        
            UPDATE record
            SET latest = false
            WHERE tag = '{self.tag}' AND combination_id = v_combination_id AND fold = '{self.fold}' AND metadata = '{self.metadata}';
            
            INSERT INTO record (tag, combination_id, fold, metadata, latest) 
            VALUES ('{self.tag}', v_combination_id, '{self.fold}', '{self.metadata}', true)
            RETURNING id INTO v_record_id;
            
            INSERT INTO record_data (record_id, result_file) 
            VALUES (v_record_id, '{self.file_storage._results_file().absolute()}')
            ON CONFLICT (record_id) DO UPDATE
            SET result_file = EXCLUDED.result_file;
  
            {self.insert_analysis(analysis)}
        END $$;
        
        COMMIT;
        """
        self.execute(query)

    def load_result(self) -> DataFrame:
        query = f"""
        BEGIN;
        
        DO $$
        DECLARE v_record_id INT;
        DECLARE v_combination_id INT;
        DECLARE v_result_file VARCHAR(1023);
        BEGIN
            SELECT id INTO v_combination_id
            FROM combination 
            WHERE model = '{self.model}' 
                AND resampler = '{self.resampler}' 
                AND scorer = '{self.scorer}'
                AND cross_validator = '{self.cv_name}';
            IF NOT FOUND THEN
                RAISE EXCEPTION 'Combination not found.';
            END IF;
            
            SELECT id INTO v_record_id
            FROM record
            WHERE tag = '{self.tag}' AND combination_id = v_combination_id;
            IF NOT FOUND THEN
                RAISE EXCEPTION 'Record not found.';
            END IF;
            
            SELECT result_file FROM record_data WHERE record_id = v_record_id INTO v_result_file;
            IF NOT FOUND THEN
                RAISE EXCEPTION 'Result file not found.';
            END IF;
            RETURN v_result_file;    
        END $$;
        COMMIT;
        """

        result = self.query(query)
        if result:
            result_file = result[0][0]
            return pd.read_csv(result_file)
        else:
            raise FileNotFoundError("Result file not found in database.")

    def save_model(self, model):
        self.file_storage.save_model(model)

        query = f"""
        BEGIN;
        
        DO $$
        DECLARE v_record_id INT;
        DECLARE v_combination_id INT;
        BEGIN
            SELECT id INTO v_combination_id
            FROM combination
            WHERE model = '{self.model}'
                AND resampler = '{self.resampler}'
                AND scorer = '{self.scorer}'
                AND cross_validator = '{self.cv_name}';
            IF NOT FOUND THEN
                INSERT INTO combination (model, resampler, scorer, cross_validator)
                VALUES ('{self.model}', '{self.resampler}', '{self.scorer}', '{self.cv_name}')
                RETURNING id INTO v_combination_id;
            END IF;
            
            IF EXISTS (
                SELECT 1 FROM record 
                WHERE tag = '{self.tag}' 
                    AND combination_id = v_combination_id
                    AND fold = '{self.fold}' 
                    AND metadata = '{self.metadata}' 
                    AND latest = true
            ) THEN
                SELECT id INTO v_record_id 
                    FROM record
                    WHERE tag = '{self.tag}'
                    AND combination_id = v_combination_id 
                    AND fold = '{self.fold}'
                    AND metadata = '{self.metadata}'
                    AND latest = true
                    LIMIT 1;
                RAISE NOTICE 'Selected ID: %', v_record_id;
            ELSE
                INSERT INTO record (tag, combination_id, fold, metadata, latest)
                VALUES ('{self.tag}', v_combination_id, '{self.fold}', '{self.metadata}', true)
                RETURNING id INTO v_record_id;
            END IF;
    
            INSERT INTO model_data (record_id, model_file) 
            VALUES (v_record_id, '{self.file_storage._model_file().absolute()}')
            ON CONFLICT (record_id) DO UPDATE
            SET model_file = EXCLUDED.model_file;
            
        END $$;
        
        COMMIT;
        """
        self.execute(query)

    def load_model(self) -> List[Model]:
        query = f"""
        BEGIN;
        
        DO $$
        DECLARE v_record_id INT;
        DECLARE v_combination_id INT;
        DECLARE v_model_file VARCHAR(1023);
        BEGIN
            SELECT id INTO v_combination_id
            FROM combination 
            WHERE model = '{self.model}' 
                AND resampler = '{self.resampler}' 
                AND scorer = '{self.scorer}'
                AND cross_validator = '{self.cv_name}';
            IF NOT FOUND THEN
                RAISE EXCEPTION 'Combination not found.';
            END IF;
            
            SELECT id INTO v_record_id
            FROM record
            WHERE tag = '{self.tag}' AND combination_id = v_combination_id;
            IF NOT FOUND THEN
                RAISE EXCEPTION 'Record not found.';
            END IF;
            
            SELECT model_file FROM model_data WHERE record_id = v_record_id INTO v_model_file;
            IF NOT FOUND THEN
                RAISE EXCEPTION 'Model file not found.';
            END IF;
            RETURN v_model_file;    
        END $$;
        COMMIT;
        """

        result = self.query(query)
        if result:
            model_file = result[0][0]
            model = pickle.load(open(model_file, "rb"))
            # assert isinstance(model, Model), "Model file found by DB is not a Model object."
            return [model]
        else:
            raise FileNotFoundError("Model file not found in database.")

    def insert_feature_set(self, feature_set: set):
        """
        This should not be called from the engine.
        It is only to be used to generate the initial feature set for a
        given combination of model, scorer, resampler.
        """
        query = f"""
        BEGIN;
        
        DO $$
        DECLARE v_combination_id INT;
        
        BEGIN
            SELECT id INTO v_combination_id
            FROM combination 
            WHERE model = '{self.model}' 
                AND resampler = '{self.resampler}' 
                AND scorer = '{self.scorer}'
                AND cross_validator = '{self.cv_name}';
            IF NOT FOUND THEN
                INSERT INTO combination (model, resampler, scorer, cross_validator) 
                VALUES ('{self.model}', '{self.resampler}', '{self.scorer}', '{self.cv_name}')
                RETURNING id INTO v_combination_id;
            END IF;
            
            INSERT INTO optimal_feature_sets (combination_id, feature_set)
            VALUES (v_combination_id, '{json.dumps(list(feature_set))}')
            ON CONFLICT (combination_id) DO UPDATE
            SET feature_set = EXCLUDED.feature_set;
        END $$;            
        """
        self.execute(query)

    def load_feature_set(self) -> list:
        query = f"""
        SELECT feature_set
        FROM optimal_feature_sets
        WHERE combination_id = (
            SELECT id
            FROM combination
            WHERE model = '{self.model}'
                AND resampler = '{self.resampler}'
                AND scorer = '{self.scorer}'
                AND cross_validator = '{self.cv_name}'
        );
        """

        result = self.query(query)
        if result:
            return list(result[0][0])
        else:
            return []

    def query(self, query) -> List[tuple]:
        cur = self._db_preaction()
        cur.execute(query)
        return cur.fetchall()

    def execute(self, query) -> None:
        cur = self._db_preaction()
        cur.execute(query)
        assert self.conn is not None
        self.conn.commit()

    @classmethod
    def _connect(cls):
        if cls.conn is None:
            cls.conn = get_connection()

    @classmethod
    def _db_preaction(cls):
        try:
            if cls.conn is None:
                cls._connect()
            assert cls.conn is not None
            cls.conn.isolation_level
            return cls.conn.cursor()
        except ps.OperationalError:
            cls._connect()
            assert cls.conn is not None
            return cls.conn.cursor()

    @classmethod
    def query_filter(cls, f: Filter) -> DataFrame:
        query = f.query()
        cls._connect()
        cur = cls._db_preaction()
        cur.execute(query)
        records = pd.read_sql(query, cls.conn)  # type: ignore

        if f.metadata is not None:
            RuntimeWarning("Metadata filtering is not implemented for DBStorage.")

        return records

    @classmethod
    def insert_analysis(cls, data: pd.DataFrame) -> str:
        assert len(data) == 1, "Only one row of data can be inserted at a time"
        columns = [
            "f1",
            "fnr",
            "fpr",
            "fp_fn",
            "tpr",
            "tnr",
            "accuracy",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "actual_positives",
            "actual_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
            "true_negatives",
            "total_predictions",
        ]

        data = data.drop(
            columns=[col for col in data.columns if col not in columns]
        )  # Drop all columns from the dataframe not expected in the database
        # print(data.columns)
        assert len(data.columns) == len(columns), "Data columns do not match expected columns"
        return f"""
            INSERT INTO analysis (record_id, {', '.join(columns)})
            VALUES (
                v_record_id,
                {', '.join([data[col].iloc[0].astype(str) for col in columns])}
            )
            ON CONFLICT (record_id) DO UPDATE
            SET {', '.join([f"{col} = EXCLUDED.{col}" for col in columns])};
        """

    def count_unique_variations(self):
        return self.execute(self.query("SELECT count(*) FROM (SELECT DISTINCT combination_id, tag FROM record) as FOO;"))

