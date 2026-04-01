# -------------------- Common training options --------------------
# Label: Parameter names
# Option: Values
# Default: When no option for label
COMMON_PARAM_CONFIG = {
    # ---------- Test size ----------
    "test_size": {
        "label": "✒️ Test Size",
        "options": {
            1: 0.1,
            2: 0.2,
            3: 0.25,
            4: 0.3,
        },
        "default": 2,
    },
    # ---------- Random state ----------
    "split_random_state": {
        "label": "🎲 Split Random State",
        "options": {
            1: 42,
            2: 0,
            3: 7,
            4: 123,
        },
        "default": 1,
    },
    # ---------- Using CV ----------
    "use_cv": {
        "label": "🧪 Use Cross Validation",
        "options": {
            1: True,
            2: False,
        },
        "default": 1,
    },
    # ---------- CV folds ----------
    "cv_folds": {
        "label": "🎰 CV Folds",
        "options": {
            1: 3,
            2: 5,
            3: 10,
        },
        "default": 2,
    },
    # ---------- Encoding options ----------
    "cat_encoder": {
        "label": "🧩 Categorical Encoder",
        "options": {
            1: "ohe",
            2: "ordinal",
        },
        "default": 1,
    },
}


# -------------------- Scoring options by task type --------------------
# Label: Parameter names
# Option: Values
# Default: When no option for label
SCORING_CONFIG = {
    # ---------- Classification scoring ----------
    "classifier": {
        "label": "🎯 Scoring",
        "options": {
            1: "accuracy",
            2: "f1",
            3: "f1_weighted",
            4: "precision_weighted",
            5: "recall_weighted",
        },
        "default": 3,
    },
    # ---------- Regression scoring ----------
    "regressor": {
        "label": "🎯 Scoring",
        "options": {
            1: "r2",
            2: "neg_mean_squared_error",
            3: "neg_mean_absolute_error",
        },
        "default": 1,
    },
}

# -------------------- Model-specific parameter options --------------------
# Label: Parameter names
# Option: Values
# Default: When no option for label
# Depend on: The label is depended on other condition
MODEL_PARAM_CONFIG = {
    # ---------- KNN Classification ----------
    "KNNClassifier": [
        {
            "name": "n_neighbors",
            "label": "🖇️ N Neighbors",
            "options": {
                1: 3,
                2: 5,
                3: 7,
                4: 9,
                5: 11,
            },
            "default": 2,
        },
        {
            "name": "algorithm",
            "label": "⚙️ Algorithm",
            "options": {
                1: "auto",
                2: "ball_tree",
                3: "kd_tree",
                4: "brute",
            },
            "default": 1,
        },
        {
            "name": "p",
            "label": "📐 Distance Power p",
            "options": {
                1: 1,
                2: 2,
            },
            "default": 2,
        },
        {
            "name": "weights",
            "label": "⚖️ Weights",
            "options": {
                1: "uniform",
                2: "distance",
            },
            "default": 1,
        },
        {
            "name": "use_pca",
            "label": "🪄 Use PCA",
            "options": {
                1: True,
                2: False,
            },
            "default": 2,
        },
        {
            "name": "pca_n_components",
            "label": "🍒 PCA Components",
            "options": {
                1: 2,
                2: 3,
                3: 5,
                4: 10,
                5: 20,
                6: None,
            },
            "default": 6,
            "depends_on": ("use_pca", True),
        },
        {
            "name": "scaler_type",
            "label": "📏 Scaler Type",
            "options": {
                1: "standard",
                2: "minmax",
                3: "robust",
                4: None,
            },
            "default": 1,
        },
    ],
    # ---------- KNN Regression ----------
    "KNNRegressor": [
        {
            "name": "n_neighbors",
            "label": "🖇️ N Neighbors",
            "options": {
                1: 3,
                2: 5,
                3: 7,
                4: 9,
                5: 11,
            },
            "default": 2,
        },
        {
            "name": "algorithm",
            "label": "⚙️ Algorithm",
            "options": {
                1: "auto",
                2: "ball_tree",
                3: "kd_tree",
                4: "brute",
            },
            "default": 1,
        },
        {
            "name": "p",
            "label": "📐 Distance Power p",
            "options": {
                1: 1,
                2: 2,
            },
            "default": 2,
        },
        {
            "name": "weights",
            "label": "⚖️ Weights",
            "options": {
                1: "uniform",
                2: "distance",
            },
            "default": 1,
        },
        {
            "name": "use_pca",
            "label": "🪄 Use PCA",
            "options": {
                1: True,
                2: False,
            },
            "default": 2,
        },
        {
            "name": "pca_n_components",
            "label": "🍒 PCA Components",
            "options": {
                1: 2,
                2: 3,
                3: 5,
                4: 10,
                5: 20,
                6: None,
            },
            "default": 6,
            "depends_on": ("use_pca", True),
        },
        {
            "name": "scaler_type",
            "label": "📏 Scaler Type",
            "options": {
                1: "standard",
                2: "minmax",
                3: "robust",
                4: None,
            },
            "default": 1,
        },
    ],
    # ---------- SVM Classification ----------
    "SVMClassifier": [
        {
            "name": "kernel",
            "label": "🧵 Kernel",
            "options": {
                1: "rbf",
                2: "linear",
                3: "poly",
                4: "sigmoid",
            },
            "default": 1,
        },
        {
            "name": "C",
            "label": "🏷️ Regularization C",
            "options": {
                1: 0.1,
                2: 1.0,
                3: 10.0,
                4: 100.0,
            },
            "default": 2,
        },
        {
            "name": "gamma",
            "label": "🧩 Gamma",
            "options": {
                1: "scale",
                2: "auto",
            },
            "default": 1,
        },
        {
            "name": "use_pca",
            "label": "🪄 Use PCA",
            "options": {
                1: True,
                2: False,
            },
            "default": 2,
        },
        {
            "name": "pca_n_components",
            "label": "🍒 PCA Components",
            "options": {
                1: 2,
                2: 3,
                3: 5,
                4: 10,
                5: 20,
                6: None,
            },
            "default": 6,
            "depends_on": ("use_pca", True),
        },
        {
            "name": "scaler_type",
            "label": "📏 Scaler Type",
            "options": {
                1: "standard",
                2: "minmax",
                3: "robust",
                4: None,
            },
            "default": 1,
        },
    ],
    # ---------- SVM Regression ----------
    "SVMRegressor": [
        {
            "name": "kernel",
            "label": "🧵 Kernel",
            "options": {
                1: "rbf",
                2: "linear",
                3: "poly",
                4: "sigmoid",
            },
            "default": 1,
        },
        {
            "name": "C",
            "label": "🏷️ Regularization C",
            "options": {
                1: 0.1,
                2: 1.0,
                3: 10.0,
                4: 100.0,
            },
            "default": 2,
        },
        {
            "name": "gamma",
            "label": "🧩 Gamma",
            "options": {
                1: "scale",
                2: "auto",
            },
            "default": 1,
        },
        {
            "name": "use_pca",
            "label": "🪄 Use PCA",
            "options": {
                1: True,
                2: False,
            },
            "default": 2,
        },
        {
            "name": "pca_n_components",
            "label": "🍒 PCA Components",
            "options": {
                1: 2,
                2: 3,
                3: 5,
                4: 10,
                5: 20,
                6: None,
            },
            "default": 6,
            "depends_on": ("use_pca", True),
        },
        {
            "name": "scaler_type",
            "label": "📏 Scaler Type",
            "options": {
                1: "standard",
                2: "minmax",
                3: "robust",
                4: None,
            },
            "default": 1,
        },
    ],
    # ---------- DecisionTree Classification ----------
    "DecisionTreeClassifier": [
        {
            "name": "criterion",
            "label": "✒️ Criterion",
            "options": {
                1: "gini",
                2: "entropy",
                3: "log_loss",
            },
            "default": 1,
        },
        {
            "name": "max_depth",
            "label": "🪵 Max Depth",
            "options": {
                1: 3,
                2: 5,
                3: 10,
                4: 20,
                5: None,
            },
            "default": 5,
        },
        {
            "name": "min_samples_split",
            "label": "🪓 Min Samples Split",
            "options": {
                1: 2,
                2: 5,
                3: 10,
            },
            "default": 1,
        },
        {
            "name": "min_samples_leaf",
            "label": "🍃 Min Samples Leaf",
            "options": {
                1: 1,
                2: 2,
                3: 4,
            },
            "default": 1,
        },
        {
            "name": "model_random_state",
            "label": "🎲 Model Random State",
            "options": {
                1: 42,
                2: 0,
                3: 7,
                4: 123,
            },
            "default": 1,
        },
    ],
    # ---------- DecisionTree Regression ----------
    "DecisionTreeRegressor": [
        {
            "name": "criterion",
            "label": "✒️ Criterion",
            "options": {
                1: "squared_error",
                2: "friedman_mse",
                3: "absolute_error",
            },
            "default": 1,
        },
        {
            "name": "max_depth",
            "label": "🪵 Max Depth",
            "options": {
                1: 3,
                2: 5,
                3: 10,
                4: 20,
                5: None,
            },
            "default": 5,
        },
        {
            "name": "min_samples_split",
            "label": "🪓 Min Samples Split",
            "options": {
                1: 2,
                2: 5,
                3: 10,
            },
            "default": 1,
        },
        {
            "name": "min_samples_leaf",
            "label": "🍃 Min Samples Leaf",
            "options": {
                1: 1,
                2: 2,
                3: 4,
            },
            "default": 1,
        },
        {
            "name": "model_random_state",
            "label": "🎲 Model Random State",
            "options": {
                1: 42,
                2: 0,
                3: 7,
                4: 123,
            },
            "default": 1,
        },
    ],
    # ---------- RandomForest Classification ----------
    "RandomForestClassifier": [
        {
            "name": "n_estimators",
            "label": "🌳 Number of Trees",
            "options": {
                1: 50,
                2: 100,
                3: 200,
            },
            "default": 2,
        },
        {
            "name": "criterion",
            "label": "✒️ Criterion",
            "options": {
                1: "gini",
                2: "entropy",
                3: "log_loss",
            },
            "default": 1,
        },
        {
            "name": "max_depth",
            "label": "🪵 Max Depth",
            "options": {
                1: 3,
                2: 5,
                3: 10,
                4: 20,
                5: None,
            },
            "default": 5,
        },
        {
            "name": "min_samples_split",
            "label": "🪓 Min Samples Split",
            "options": {
                1: 2,
                2: 5,
                3: 10,
            },
            "default": 1,
        },
        {
            "name": "min_samples_leaf",
            "label": "🍃 Min Samples Leaf",
            "options": {
                1: 1,
                2: 2,
                3: 4,
            },
            "default": 1,
        },
        {
            "name": "model_random_state",
            "label": "🎲 Model Random State",
            "options": {
                1: 42,
                2: 0,
                3: 7,
                4: 123,
            },
            "default": 1,
        },
    ],
    # ---------- RandomForest Regression ----------
    "RandomForestRegressor": [
        {
            "name": "n_estimators",
            "label": "🌳 Number of Trees",
            "options": {
                1: 50,
                2: 100,
                3: 200,
            },
            "default": 2,
        },
        {
            "name": "criterion",
            "label": "✒️ Criterion",
            "options": {
                1: "squared_error",
                2: "friedman_mse",
                3: "absolute_error",
            },
            "default": 1,
        },
        {
            "name": "max_depth",
            "label": "🪵 Max Depth",
            "options": {
                1: 3,
                2: 5,
                3: 10,
                4: 20,
                5: None,
            },
            "default": 5,
        },
        {
            "name": "min_samples_split",
            "label": "🪓 Min Samples Split",
            "options": {
                1: 2,
                2: 5,
                3: 10,
            },
            "default": 1,
        },
        {
            "name": "min_samples_leaf",
            "label": "🍃 Min Samples Leaf",
            "options": {
                1: 1,
                2: 2,
                3: 4,
            },
            "default": 1,
        },
        {
            "name": "model_random_state",
            "label": "🎲 Model Random State",
            "options": {
                1: 42,
                2: 0,
                3: 7,
                4: 123,
            },
            "default": 1,
        },
    ],
}


# =================================================
