import sweetviz as sv
import pandas as pd


def generate_report(data: pd.DataFrame, target: str = "definitive_diagnosis", open_browser: bool = False):
    """
    Generate statistics and reports based on the given dataset using Sweetviz.
    Save the reports to the current directory.

    :param data: The dataset to generate reports from.
    :param target: The target feature to analyze.
    """

    # Split into two datasets based on definitive_diagnosis
    positive = data[data[target] == 1]
    negative = data[data[target] == 0]

    # Generate reports
    report = sv.compare([positive, "Positive"], [negative, "Negative"]) # type: ignore
    report.show_html("./compare_dataset.html", open_browser=False)

    report = sv.analyze(data, target_feat="definitive_diagnosis", pairwise_analysis="on")
    report.show_html("./dataset.html", open_browser=open_browser)
