import numpy as np
import pickle
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# Utility: Save best hyperparameters
def save_best_params(params, file_path):
    print(f"Saving best parameters to {file_path}")
    with open(file_path, 'wb') as file:
        pickle.dump(params, file)
    print("Parameters saved successfully!")

# Utility: Calculate Gini coefficient
def calculate_gini(y_true, y_pred_proba):
    gini_score = 2 * roc_auc_score(y_true, y_pred_proba) - 1
    return gini_score

# Function: Train Logistic Regression
def train_logistic_regression(X_train, y_train, X_val=None, y_val=None, param_file=None, max_evals=50):
    X_val = X_val if X_val is not None else X_train
    y_val = y_val if y_val is not None else y_train

    def objective(params):
        try:
            # Validate incompatible parameter combinations
            if (params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']) or \
               (params['penalty'] == 'elasticnet' and params['solver'] != 'saga') or \
               (params['dual'] and params['solver'] != 'liblinear'):
                raise ValueError(f"Invalid combination: {params}")

            l1_ratio = params['l1_ratio'] if params['penalty'] == 'elasticnet' else None

            model = LogisticRegression(
                C=params['C'],
                penalty=params['penalty'],
                solver=params['solver'],
                max_iter=int(params['max_iter']),
                dual=params['dual'],
                class_weight=params['class_weight'],
                l1_ratio=l1_ratio,
                random_state=42
            )

            auc = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1).mean()
            return {'loss': -auc, 'status': STATUS_OK}

        except Exception as e:
            print(f"Error during evaluation: {e}, Params: {params}")
            return {'loss': float('inf'), 'status': STATUS_OK}

    # Define the hyperparameter search space
    space = {
        'C': hp.loguniform('C', np.log(1e-4), np.log(1e1)),
        'penalty': hp.choice('penalty', ['l1', 'l2', 'elasticnet']),
        'solver': hp.choice('solver', ['liblinear', 'saga', 'lbfgs']),
        'max_iter': hp.quniform('max_iter', 100, 5000, 50),
        'dual': hp.choice('dual', [False, True]),
        'class_weight': hp.choice('class_weight', [None, 'balanced']),
        'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0)
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    # Extract best parameters
    best_params = {
        'C': best['C'],
        'penalty': ['l1', 'l2', 'elasticnet'][best['penalty']],
        'solver': ['liblinear', 'saga', 'lbfgs'][best['solver']],
        'max_iter': int(best['max_iter']),
        'dual': bool(best['dual']),
        'class_weight': [None, 'balanced'][best['class_weight']],
        'l1_ratio': best['l1_ratio'] if ['l1', 'l2', 'elasticnet'][best['penalty']] == 'elasticnet' else None
    }

    model = LogisticRegression(**best_params, random_state=42)
    model.fit(X_train, y_train)
    val_pred_proba = model.predict_proba(X_val)[:, 1]

    if param_file:
        save_best_params(best_params, param_file)

    return model, best_params, val_pred_proba


def train_decision_tree(X_train, y_train, X_val=None, y_val=None, param_file=None, max_evals=50):
    X_val = X_val if X_val is not None else X_train
    y_val = y_val if y_val is not None else y_train

    # Objective function for hyperopt
    def objective(params):
        model_params = {
            'max_depth': int(params['max_depth']),
            'min_samples_split': int(params['min_samples_split']),
            'min_samples_leaf': int(params['min_samples_leaf']),
            'criterion': params['criterion'],
            'splitter': params['splitter'],
            'min_impurity_decrease': params['min_impurity_decrease']
        }
        model = DecisionTreeClassifier(**model_params, random_state=42)
        auc = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1).mean()
        return {'loss': -auc, 'status': STATUS_OK}

    # Hyperparameter search space
    space = {
    'max_depth': hp.quniform('max_depth', 5, 50, 1),  
    'min_samples_split': hp.quniform('min_samples_split', 5, 100, 5),  
    'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 50, 1), 
    'criterion': hp.choice('criterion', ['gini', 'entropy']), 
    'splitter': hp.choice('splitter', ['best', 'random']),
    'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 0.1)  
    }

    # Run optimization
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    # Train final model
    best_params = {
        'max_depth': int(best['max_depth']),
        'min_samples_split': int(best['min_samples_split']),
        'min_samples_leaf': int(best['min_samples_leaf']),
        'criterion': ['gini', 'entropy'][best['criterion']]
    }
    model = DecisionTreeClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    val_pred_proba = model.predict_proba(X_val)[:, 1]

    # Save best parameters
    if param_file:
        save_best_params(best_params, param_file)

    return model, best_params, val_pred_proba