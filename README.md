# Zeus-Supervised-ML

A modular terminal-based supervised machine learning system for classification, regression, model evaluation, visualization, and trained-model management.

Zeus-Supervised-ML is a Python-based supervised machine learning project designed as an interactive terminal system.  
It provides a structured workflow for loading datasets, selecting features and targets, training models, evaluating performance, visualizing results, and saving or reloading trained models.

This project was built not only to practice machine learning, but also to demonstrate modular project architecture, reusable design, and engineering-oriented workflow development.

---

## Project Overview

Zeus is a menu-driven supervised learning framework that separates machine learning workflows into organized modules.  
The project emphasizes both practical model usage and clean system design.

It includes:

- dataset loading and preprocessing workflow
- feature and target selection
- classification and regression model training
- model evaluation and visualization
- modular engine / mission / model / config architecture
- model saving and loading support
- workflow logging

---

## Features

- Interactive terminal menu system
- Load and manage machine learning datasets
- Select target column(s) and feature column(s)
- Automatically use all non-target columns when feature columns are not manually specified
- Support both **classification** and **regression** tasks
- Support both **single-output** and **multi-output** supervised learning workflows
- Automatically encode non-numeric classification targets (`y`) when needed
- Preserve original class-label names for classification evaluation display when available
- Support per-target handling for multi-output classification workflows
- Modular engine-based workflow design
- Built-in evaluation, diagnostic, and plotting functions
- Support train/test selection for ROC and Precision-Recall plots
- Save and reload trained models
- Logging support for workflow tracking and debugging
- Optional cross-validation training with GridSearchCV
- Store compact CV search summaries and raw CV search results
- Export top-ranked CV search results as CSV reports
- Export evaluation results as text reports
- Menu-level validation for known invalid preprocessing combinations

---

## Supported Models

### Classification
- Decision Tree Classifier
- Random Forest Classifier
- SVM Classifier
- KNN Classifier

### Regression
- Decision Tree Regressor
- Random Forest Regressor
- SVM Regressor
- KNN Regressor

---

## Classification Target Handling

For classification workflows, Zeus supports automatic target-label encoding when the selected target column contains non-numeric class labels.

This means Zeus can handle classification targets such as:

- `"yes"` / `"no"`
- `"spam"` / `"ham"`
- `"A"` / `"B"` / `"C"`

without requiring manual preprocessing by the user.

### Supported behavior
- Single-output classification target encoding
- Multi-output classification target encoding
- Per-target label encoding for multi-output classification
- Inverse-transform display for prediction previews
- Original class names preserved in classification reports when available

### Notes
- Target encoding is applied only to **classification** workflows
- Numeric and boolean targets are kept unchanged
- Regression targets are not label-encoded

---

## Evaluation and Visualization

Depending on the selected model and task type, Zeus supports evaluation and visualization features such as:

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Decision Function Distribution (SVC classification)
- Tree Plot
- Regression Diagnostics
- Residual-related plots
- Feature Importance (supported models only)

For classification plotting workflows, Zeus also supports selecting either the training split or the testing split when generating ROC and Precision-Recall curves.

For multi-output classification workflows, Zeus supports selecting a specific target column for plotting tasks that require a single-target view.

These tools help users better understand model performance, compare classification behavior across dataset splits, and interpret prediction results more clearly.

---

### Binary-only plot note
Some classifier diagnostic plots currently support **binary classification only**, including:

- ROC Curve
- Precision-Recall Curve
- Decision Function Distribution (SVC classification)

If the selected target contains more than two classes, Zeus will display a user-facing message and stop the plotting workflow gracefully instead of treating it as a system error.

---

## Training Workflow Notes

Zeus includes menu-level validation to block known invalid preprocessing combinations before training begins.

For example, in KNN / SVM family workflows, a combination such as:

- `cat_encoder = "ohe"`
- `scaler_type = "standard"`

may be blocked in the menu layer because one-hot encoding commonly produces sparse output, while standard scaling with default centering behavior may fail on sparse matrices in this workflow.

This validation is intended to prevent avoidable runtime errors and guide users toward safer parameter combinations.

In addition, Zeus distinguishes between cancelled menu input and valid parameter values such as `None`, so options like `scaler_type = None` can be preserved correctly during training parameter collection.

---

## Project Structure

```bash
Zeus-Supervised-ML/
│
├── ML_BaseConfigBox/
├── ML_MissionBox/
├── ML_ModelBox/
│
├── ML_Report/
├── Menu_Config.py
├── Menu_Helper_Decorator.py
├── Zeus_Logging.py
├── Zeus_ML_Engine.py
├── Zeus_Main.py
├── Zeus_Menu1.py
├── Zeus_Menu2.py
├── Zeus_Menu3.py
├── Zeus_Model_Menu_Helper.py
├── requirements.txt
└── .gitignore