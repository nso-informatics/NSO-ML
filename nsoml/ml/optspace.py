from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, RUSBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_MODELS = {
    "RandomForest": {
        "estimator": RandomForestClassifier(),
        "search_spaces": {
            "bootstrap": Categorical([True, False]),
            "criterion": Categorical(["gini", "entropy"]),
            "max_depth": Integer(3, 50),
            "max_features": Categorical(["sqrt", "log2"]),
            "min_samples_leaf": Integer(1, 10),
            "min_samples_split": Integer(2, 10),
            "n_estimators": Integer(10, 500, prior="log-uniform"),
        },
    },
    "RandomForest(balanced)": {
        "estimator": RandomForestClassifier(class_weight="balanced"),
        "search_spaces": {
            "bootstrap": Categorical([True, False]),
            "criterion": Categorical(["gini", "entropy"]),
            "max_depth": Integer(3, 50),
            "max_features": Categorical(["sqrt", "log2"]),
            "min_samples_leaf": Integer(1, 10),
            "min_samples_split": Integer(2, 10),
            "n_estimators": Integer(10, 500, prior="log-uniform"),
        },
    },
    "BalancedBagging": {
        "estimator": BalancedBaggingClassifier(),
        "search_spaces": {
            "warm_start": Categorical([True, False]),
            "bootstrap": Categorical([True, False]),
            "n_estimators": Integer(5, 50),
#            "sampling_strategy": Real(RATIO, 1, prior="log-uniform"), # Controls sampling rate for RUS.
        },
    },
    "BalancedRandomForest": {
        "estimator": BalancedRandomForestClassifier(),
        "search_spaces": {
            "criterion": Categorical(["gini", "entropy"]),
            "max_depth": Integer(3, 50),
            "max_features": Categorical(["sqrt", "log2"]),
            "min_samples_leaf": Integer(1, 10),
            "min_samples_split": Integer(2, 10),
            "n_estimators": Integer(50, 500),
#            "sampling_strategy": Real(RATIO, 1, prior="log-uniform"), # Controls sampling rate for RUS.
        },
    },
    "BalancedRandomForest(balanced)": {
        "estimator": BalancedRandomForestClassifier(class_weight="balanced"),
        "search_spaces": {
            "criterion": Categorical(["gini", "entropy"]),
            "max_depth": Integer(3, 50),
            "max_features": Categorical(["sqrt", "log2"]),
            "min_samples_leaf": Integer(1, 10),
            "min_samples_split": Integer(2, 10),
            "n_estimators": Integer(50, 500),
#            "sampling_strategy": Real(RATIO, 1, prior="log-uniform"), # Controls sampling rate for RUS.
        },
    },
    "GradientBoosting": {
        "estimator": XGBClassifier(),
        "search_spaces": {
            "max_depth": Integer(1, 10),
            "gamma": Real(0.1, 10, prior="log-uniform"),
            "subsample": Real(0.5, 1, prior="log-uniform"),
            "min_child_weight": Integer(1, 10),
            "colsample_bytree": Real(0.5, 1, prior="log-uniform"),
            "learning_rate": Real(0.1, 1, prior="log-uniform"),
            "max_delta_step": Integer(0, 10),
            "lambda": Integer(1, 3),
            "alpha": Integer(0, 2),
        },
    },
    "RUSBoost": {
        "estimator": RUSBoostClassifier(),
        "search_spaces": {
            "learning_rate": Real(0.1, 1, prior="log-uniform"),
            "n_estimators": Integer(10, 500),
 #           "sampling_strategy": Real(RATIO, 1, prior="log-uniform"), # Controls sampling rate for RUS.
        },
    },
#   "LinearSVC": {
#       "estimator": make_pipeline(
#           StandardScaler(), LinearSVC(max_iter=100000, dual=False)
#       ),
#       "search_spaces": {
#           "linearsvc__loss": Categorical(["squared_hinge"]),
#           "linearsvc__C": Real(0.1, 100, prior="log-uniform"),
#       },
#   }, 
#   "LinearSVC(balanced)": {
#       "estimator": make_pipeline(
#           StandardScaler(), LinearSVC(class_weight="balanced", max_iter=100000, dual=False)
#       ),
#       "search_spaces": {
#           "linearsvc__loss": Categorical(["squared_hinge"]),
#           "linearsvc__C": Real(0.1, 100, prior="log-uniform"),
#       },
#   }, 
}
