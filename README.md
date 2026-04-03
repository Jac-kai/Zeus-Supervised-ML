# Zeus-Supervised-ML

A modular terminal-based supervised machine learning system for classification, regression, model evaluation, and visualization.

Zeus-Supervised-ML is a Python-based supervised machine learning project designed as an interactive terminal system.  
It provides a structured workflow for loading datasets, selecting features and targets, training models, evaluating performance, visualizing results, and saving trained models.

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
- Modular engine-based workflow design
- Built-in evaluation, diagnostic, and plotting functions
- Support train/test selection for ROC and Precision-Recall plots
- Save and reload trained models
- Logging support for workflow tracking and debugging

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

## Evaluation and Visualization

Depending on the selected model and task type, Zeus supports evaluation and visualization features such as:

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Decision Function Distribution (SVC classification)
- Tree Plot
- Regression Diagnostics
- Residual-related plots

For classification plotting workflows, Zeus also supports selecting either the training split or the testing split when generating ROC and Precision-Recall curves.

These tools help users better understand model performance, compare classification behavior across dataset splits, and interpret prediction results more clearly.

---

## Project Structure

```bash
Zeus-Supervised-ML/
│
├── ML_BaseConfigBox/
├── ML_MissionBox/
├── ML_ModelBox/
│
├── Menu_Config.py
├── Menu_Helper_Decorator.py
├── Zeus_Logging.py
├── Zeus_ML_Engine.py
├── Zeus_Main.py
├── Zeus_Menu1.py
├── Zeus_Menu2.py
├── Zeus_Menu3.py
├── Zeus_Model_Menu_Helper.py
└── .gitignore
```

### Main Module Description

- **Zeus_Main.py**  
  Entry point of the project. Runs the main terminal workflow.

- **Zeus_ML_Engine.py**  
  Core engine of the Zeus system. Handles workflow control, model execution, and engine-level operations.

- **Zeus_Menu1.py**  
  Menu for dataset loading and feature/target selection.

- **Zeus_Menu2.py**  
  Menu for model configuration, model training, and model save/load operations.

- **Zeus_Menu3.py**  
  Menu for evaluation, plotting, and result visualization.

- **Zeus_Logging.py**  
  Logging setup for recording workflow activity and debugging information.

- **ML_ModelBox/**  
  Contains model-layer implementations.

- **ML_MissionBox/**  
  Contains mission/task-layer workflow logic.

- **ML_BaseConfigBox/**  
  Contains shared base configurations and reusable ML pipeline utilities.

---

## Workflow

A typical Zeus workflow is:

1. Load dataset
2. Select target column(s)
3. Select feature column(s), or use all non-target columns
4. Choose a supervised learning model
5. Train the model
6. Evaluate the model
7. Visualize the results
8. Save the trained model

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Jac-kai/Zeus-Supervised-ML.git
cd Zeus-Supervised-ML
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### Windows
```bash
.venv\Scripts\activate
```

### macOS / Linux
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, you may install the main libraries manually:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

---

## How to Run

Run the main program:

```bash
python Zeus_Main.py
```

Then follow the terminal menu instructions to complete the supervised learning workflow.

---

## Project Goals

This project was built to:

- practice supervised machine learning workflow design
- improve Python software structuring ability
- develop reusable ML system architecture
- demonstrate hands-on skills in model training, evaluation, and workflow engineering

---

## Why This Project Matters

This project demonstrates not only machine learning model usage, but also software engineering thinking through:

- modular architecture
- reusable components
- workflow-oriented system design
- organized project layering for maintainability and extensibility

It reflects my effort to transition into AI and data-related roles by combining machine learning practice with practical software development skills.

---

## Future Improvements

Possible future extensions include:

- adding more supervised learning models
- supporting hyperparameter tuning workflows
- exporting evaluation reports automatically
- adding more visualization options
- improving saved model reload workflows
- including example datasets and screenshots
- adding model performance comparison reports

---

## Author

**Zack**  
GitHub: [Jac-kai](https://github.com/Jac-kai)

---

## License

This project is for learning, portfolio, and demonstration purposes.