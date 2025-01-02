from setuptools import setup, find_packages

setup(
    name="nsoml",
    version="0.0.1",
    description="A Python package for machine learning with NSO data",
    author="Alexander De Furia",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scikit-optimize",
        "imbalanced-learn",
        "matplotlib",
        "seaborn",
        "jupyter",
        "joblib",
        "tqdm",
        "xgboost",
        "openpyxl",
        "sweetviz",
        "pyright",
        "fastparquet",
        "pyarrow",
    ],
    packages=find_packages(include=["nsoml", "nsoml.*", "*"]),
)
