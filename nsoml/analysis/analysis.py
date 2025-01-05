import datetime
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import re
import time
from typing import Dict, List, Optional, Pattern, Tuple

import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from sklearn.base import BaseEstimator as Model
from ..storage.db_connection import get_connection

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import warnings


class Filter:
    def __init__(self, 
                 cross_validator: Optional[str] = None,
                 model: Optional[str] = None, 
                 resampler: Optional[str] = None, 
                 score: Optional[str] = None, 
                 tag: Optional[str] = None,
                 combination_id: Optional[int] = None,
                 metadata: dict = {}):
        self.cross_validator: Pattern = re.compile(cross_validator) if cross_validator else re.compile(r'(.*)')
        self.model: Pattern = re.compile(model) if model else re.compile(r'(.*)')
        self.resampler = re.compile(resampler) if resampler else re.compile(r'(.*)') 
        self.score = re.compile(score) if score else re.compile(r'(.*)')
        self.tag = re.compile(tag) if tag else re.compile(r'(.*)')
        self.combination_id = combination_id
        self.metadata = metadata
        
    def query(self):
        return f"""
            SELECT * FROM record
            INNER JOIN combination ON record.combination_id = combination.id
            INNER JOIN analysis ON record.id = analysis.record_id
            INNER JOIN model_data ON record.id = model_data.record_id
            INNER JOIN record_data ON record.id = record_data.record_id
            WHERE cross_validator ~ '{self.cross_validator.pattern}'
            AND model ~ '{self.model.pattern}'
            AND resampler ~ '{self.resampler.pattern}'
            AND scorer ~ '{self.score.pattern}'
            AND tag ~ '{self.tag.pattern}'
            AND latest = TRUE
            {f"AND combination.id = {self.combination_id}" if self.combination_id else ""};
        """

class ResultSource:
    def __init__(self, 
                 cross_validator_dir: Path, 
                 resampler_dir: Path, 
                 model_dir: Path, 
                 score_dir: Path, 
                 file: Path, 
                 tag="no_label"):
        self.cross_validator_dir = cross_validator_dir
        self.resampler_dir = resampler_dir
        self.model_dir = model_dir
        self.score_dir = score_dir
        self.file = file
        self.tag = tag
        
    def __iter__(self):
        return iter([self.cross_validator_dir, 
                     self.resampler_dir, 
                     self.model_dir, 
                     self.score_dir, 
                     self.file, 
                     self.tag])

    def __str__(self) -> str:
        names = [Path(x).name for x in self.__iter__()]
        return "_".join(names)

    def __repr__(self) -> str:
        return self.__str__()
    

class Analysis:
    def __init__(self, 
                 load_models: bool = False, 
                 load_dataframes: bool = False, 
                 filter: Filter = Filter(), 
                 output: bool = True, 
                 records_path: Path = Path('records'),
                 plot_roc: bool = False,
                 overwrite_plots: bool = False):
        self.records_path: Path = records_path
        self.report_path: Path = Path('reports')
        self.raw_data: pd.DataFrame = pd.DataFrame()
        self.models: Dict[str, Model] = {}
        self.results: pd.DataFrame = pd.DataFrame()
        self.analytics_file: Path = Path('')
        self.load_models: bool = load_models
        self.load_dataframes: bool = load_dataframes
        self.filter: Filter = filter
        self.results_csv_regex = re.compile(r"results_cv_(\d+)\.csv")
        self.output: bool = output
        self.plot_roc: bool = plot_roc
        self.overwrite_plots: bool = overwrite_plots
        self.job_list: List[ResultSource] = self._discover_files()     

    def load_data(self) -> None:
        """
        Load the data from the records directory and calculate the metrics.
        """
        # Load the data
        for job in tqdm(self.job_list, desc="Loading data"):
            try:
                self._handle_file(job)
                if not self.load_dataframes:
                    print("Not loading dataframes")
                    self.raw_data = pd.DataFrame() # Clear the data to save memory and processing time
            except Exception as e:
                print(f"Error processing {job}")
                raise e
        
        # Save the analysis
        if self.output: 
            self._save()
    
    def save_analysis_db(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conn = get_connection()
            query = self.filter.query()
            self.results = pd.read_sql(query, conn) # type: ignore
            self._save()
        return self.results
        
    def create_plots(self, overwrite_plots: bool = False) -> None:
        # Plot ROC curves
        if self.overwrite_plots or overwrite_plots:
            for source in tqdm(self.job_list, desc="Deleting old ROC curves"):
                # Clear old ROC plots
                results_dir = source.file.parent
                for x in results_dir.iterdir():
                    if x.suffix == '.png': x.unlink()
            
        for source in tqdm(self.job_list, desc="Plotting ROC curves"):
            results_dir = source.file.parent
            # Check if we have alreay created plots (.png)
            if any([x.suffix == '.png' for x in results_dir.iterdir()]): continue
            csv_result_files = [x for x in results_dir.iterdir() if x.suffix == '.csv']
            if len(csv_result_files) == 0: continue
            
            # Join all the results into a single dataframe
            df = pd.concat([pd.read_csv(x) for x in csv_result_files])
            self.run_roc(df, results_dir, source) 


    def _discover_files(self) -> List[ResultSource]:
        job_list: List[ResultSource] = []
        for cross_validator_dir in self.records_path.iterdir():
            if not self.filter.cross_validator.match(cross_validator_dir.name): continue
            if not cross_validator_dir.is_dir(): continue
            for resampler_dir in cross_validator_dir.iterdir():
                if not self.filter.resampler.match(resampler_dir.name): continue
                if not resampler_dir.is_dir(): continue
                for model_dir in resampler_dir.iterdir():
                    if not self.filter.model.match(model_dir.name): continue
                    if not model_dir.is_dir(): continue
                    for score_dir in model_dir.iterdir():
                        if not self.filter.score.match(score_dir.name): continue
                        if not score_dir.is_dir(): continue
                        for tag_dir in score_dir.iterdir():
                            if not self.filter.tag.match(tag_dir.name): continue
                            if not tag_dir.is_dir(): continue
                            timestamps = [x for x in tag_dir.iterdir() if x.is_dir()]
                            timestamps.sort(key=lambda x: time.mktime(time.strptime(x.name, "%Y-%m-%d_%H-%M-%S")), reverse=True)
                            for record in timestamps[0].iterdir():
                                if not record.is_file(): continue                                
                                job_list.append(ResultSource(cross_validator_dir, 
                                                             resampler_dir, 
                                                             model_dir, 
                                                             score_dir, 
                                                             tag_dir, 
                                                             record.name))

        return job_list
    
    # def _handle_file(self, cross_validator_dir: Path, resampler_dir: Path, model_dir: Path, score_dir: Path, file: Path, tag = "") -> None:
    def _handle_file(self, source: ResultSource) -> None:
        print(f"Processing {source}")
        if source.file.suffix == '.csv':
            record_data = pd.read_csv(source.file)
            record_data['cross_validator'] = source.cross_validator_dir.name
            record_data['resampler'] = source.resampler_dir.name
            record_data['model'] = source.model_dir.name
            record_data['score'] = source.score_dir.name
            record_data['tag'] = source.tag
            match = self.results_csv_regex.match(source.file.name)
            record_data['fold'] = int(match.group(1)) if match else -1
            self.raw_data = pd.concat([self.raw_data, record_data])
            print(f"Loaded {source.file.name}")
            # Run the metric calculations
            self.results = pd.concat([self.results, self.calculate_metrics(record_data)])

        elif '_model.pkl' in source.file.name and self.load_models: 
            self.models[str(source)] = pickle.load(open(source.file, 'rb')) # type: ignore
 
    @classmethod
    def calculate_metrics(cls, group: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various metrics based on the provided group DataFrame.

        Parameters:
        - group (pd.DataFrame): The DataFrame containing the data for which metrics need to be calculated.

        Returns:
        - None
        """

        def safe_divide(a, b, default=0) -> float:
            return a / b if b != 0 else default

        # Count!
        actual = group['actual']
        predicted = group['predicted']
        total_predictions = len(group)
        true_positives = ((actual == 1) & (predicted == 1)).sum()
        false_positives = ((actual == 0) & (predicted == 1)).sum()
        false_negatives = ((actual == 1) & (predicted == 0)).sum()
        true_negatives = ((actual == 0) & (predicted == 0)).sum()
        
        # Calculate the metrics
        f1 = f1_score(actual, predicted)
        actual_positives = true_positives + false_negatives
        actual_negatives = true_negatives + false_positives
        sensitivity = safe_divide(true_positives, actual_positives)
        specificity = safe_divide(true_negatives, actual_negatives)
        accuracy = safe_divide(true_positives + true_negatives, total_predictions)
        ppv = safe_divide(true_positives, true_positives + false_positives)
        npv = safe_divide(true_negatives, true_negatives + false_negatives)
        fnr = safe_divide(false_negatives, actual_positives)
        fpr = safe_divide(false_positives, actual_negatives)
        tpr = safe_divide(true_positives, actual_positives)
        tnr = safe_divide(true_negatives, actual_negatives)
        fp_fn = safe_divide(1, fpr + fnr, default=100)

        # Create a new DataFrame with calculated metrics
        return pd.DataFrame({
            # Metadata
            'cross_validator': group['cross_validator'].unique(),
            'model': group['model'].unique(),
            'resampler': group['resampler'].unique(),
            'score': group['score'].unique(),
            'tag': group['tag'].unique(),
            'fold': group['fold'].unique(),
            # Metrics
            'f1': f1,
            'fnr': fnr,
            'fpr': fpr,
            'fp_fn': fp_fn,
            'tpr': tpr,
            'tnr': tnr,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'sensitivity': sensitivity,
            'actual_positives': actual_positives,
            'actual_negatives': actual_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'total_predictions': total_predictions
        })

    def _save(self) -> None:
            """
            Save the analysis results to a CSV file.

            The analysis results are saved in a directory named with the current date and time.
            The CSV file is named 'analysis.csv' and is saved within the directory.

            Returns:
                None
            """
            directory = self.report_path / Path("analysis") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(directory)
            self.analytics_file = directory / "analysis.csv"
            self.results.to_csv(self.analytics_file, index=False)
        
    def run_roc(self, df: pd.DataFrame, output_path: Path, result_source: Optional[ResultSource] = None) -> None:
            """
            Generate and save ROC curve and threshold plots based on the given DataFrame.

            Parameters:
            - df (pd.DataFrame): DataFrame containing the actual, predicted, and proba columns.
            - output_path (Path): Path to save the generated plots.
            - result_source (Optional[ResultSource]): Optional parameter to provide additional information for the plot titles.

            Returns:
            - None
            """
            
            required_columns = ['actual', 'predicted', 'proba']
            if "proba" not in df.columns:
                return
            assert all([x in df.columns for x in required_columns]), f"ROC curve requires {required_columns}"
            assert len(df['actual'].unique()) == 2, "ROC curve requires binary classification"
            assert len(df['predicted'].unique()) == 2, "ROC curve requires binary classification"
            
            permutation_name = None
            if result_source:
                permutation_name = result_source.cross_validator_dir.name + ', '
                if result_source.resampler_dir.name != 'none':
                    permutation_name += result_source.resampler_dir.name + ', '
                permutation_name += result_source.model_dir.name + ', '
                permutation_name += result_source.score_dir.name + ', '
                if result_source.tag != 'base':
                    permutation_name += result_source.tag + ', '
                permutation_name = permutation_name[:-2] # Remove the last comma and space
        
            our_threshold = 0.5
            fpr, tpr, thresholds = roc_curve(df['actual'], df['proba'])
            roc_auc = auc(fpr, tpr)
            closest_threshold_index = (np.abs(thresholds - our_threshold)).argmin()
            fpr_for_threshold = fpr[closest_threshold_index]

            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
            plt.plot([fpr_for_threshold, fpr_for_threshold], [0, 1], linestyle='--', color='red', lw=1, label=f'Evaluated at FPR = {fpr_for_threshold:.4f}')
            plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            if result_source and permutation_name:
                plt.suptitle('ROC Curve', fontsize=16, y=0.95)    
                plt.title(permutation_name, fontsize=8, pad=10)
            else:
                plt.title('ROC Curve', fontsize=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
            plt.tight_layout()
            plt.savefig(output_path / 'roc_curve.png')
            plt.close()

            # PLot AUROC
            plt.figure(figsize=(5, 4))
            plt.plot(thresholds, fpr, color='blue', lw=2, label='FPR')
            plt.plot(thresholds, tpr, color='green', lw=2, label='TPR')
            plt.plot([our_threshold, our_threshold], [0, 1], color='red', lw=1, linestyle='--', label=f'Evaluated at FPR = {fpr_for_threshold:.2f}')
            plt.xlabel('Performance by Decision Threshold')
            plt.ylabel('Rate')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            if result_source and permutation_name:
                plt.suptitle('Thresholds', fontsize=16, y=0.95)    
                plt.title(permutation_name, fontsize=8, pad=10)
            else:
                plt.title('Thresholds', fontsize=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
            plt.tight_layout()
            plt.savefig(output_path / 'thresholds.png')
            plt.close()
        
    def collect_plot_paths(self, zip_files=False) -> List[Path] | Path:
        """
        Collect all the plot paths and return them as a list of paths or a zip file.

        Args:
            zip (bool, optional): If True, the plot paths will be returned as a zip file. 
                                  If False (default), the plot paths will be returned as a list of paths.
        Returns:
            List[Path] | Path: If `zip` is True, returns the path to the generated zip file.
                               If `zip` is False, returns a list of paths to the collected plot files.
        """
        
        parent_dir = set([x.file.parent for x in self.job_list])
        paths: List[Path] = []
        for directory in parent_dir:
            for file in directory.iterdir():
                if file.suffix == '.png':
                    paths.append(file)
                    
        if zip_files:
            import zipfile
            self.plots_zip_file = self.report_path / "plots.zip"
            with zipfile.ZipFile(self.plots_zip_file, 'w') as z:
                for file in tqdm(paths, desc="Zipping plots"):
                    z.write(file)
                    
            return self.plots_zip_file
        
        return paths


