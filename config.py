import os


# --- Run mode ---

IS_TRAIN = False
LOCAL = True
IS_RERUN = True

# --- Config ---

class Config:
    seed = 42

    is_train = IS_TRAIN
    use_min_max = False
    use_std_scaler = True


    path_to_train = "/home/jollianreap/python_projects/titanic_project/titanic/train.csv" if LOCAL else ''
    path_to_test = "/home/jollianreap/python_projects/titanic_project/titanic/test.csv" if LOCAL else ''
    path_to_submission = "/home/jollianreap/python_projects/titanic_project/titanic/gender_submission.csv" if LOCAL else ''

    path_to_save_data_checkpoint = 'checkpoints/data_checkpoint_base'     # Drop columns, categorical columns, etc.
    path_to_save_solver_checkpoint = 'checkpoints/solver_checkpoint_base' # Models, weights, etc.

    path_to_checkpoints = 'checkpoints/' if LOCAL else '/kaggle/input/mcts-solution-checkpoint/'
    path_to_load_data_checkpoint = os.path.join(path_to_checkpoints, 'data_checkpoint.pickle')
    path_to_load_solver_checkpoint = {
        "log_reg": path_to_save_solver_checkpoint,
        "ridge": path_to_save_solver_checkpoint,
        "lasso": path_to_save_solver_checkpoint,
        "elastic": path_to_save_solver_checkpoint,
        "knn": path_to_save_solver_checkpoint,
        "tree": path_to_save_solver_checkpoint,
        "random_forest": path_to_save_solver_checkpoint,
        "catboost": path_to_save_solver_checkpoint,
        "lgbm": path_to_save_solver_checkpoint,
        "xgboost": path_to_save_solver_checkpoint,
        'DNN': path_to_save_solver_checkpoint
    }

    path_to_save_dataset = 'checkpoints/dataset.csv'
    load_dataset_checkpoint = False

    task = "classification"

    n_splits = 5

    use_oof = False
    use_dnn_embeddings = True
    use_baseline_scores = False
    show_shap = False
    mask_filter = False
    stacked = False

    log_reg_params = {
        'penalty': 'l1',
        'max_iter': 2000,
        'solver': 'liblinear',
        'C': 2
    }
    ridge_params = {
        'alpha': 0.5,
        'max_iter': 1200,
        'solver': 'svd',
    }

    knn_params = {
        'n_neighbors': 10,
        'weights': 'distance',
        'algorithm': 'brute',
        'p': 1.4,
        }
    tree_params = {
        'criterion': 'log_loss',
        'max_depth': 20,
        'min_samples_leaf': 5,
        'max_features': 'log2',
    }
    random_forest_params = {
        'n_estimators': 300,
        'criterion': 'log_loss',
        'max_features': 'log2',
        'min_samples_split': 5,
        'max_depth': 20,
        'min_impurity_decrease': 0.00001
    }

    catboost_params = {
        'iterations': 1000,
        'learning_rate': 0.01,
        'l2_leaf_reg': 1.5,
        'bootstrap_type': 'Bernoulli',
        'logging_level': 'Silent'
    }
    lgbm_params = {
        'early_stopping_rounds': 0,
        'verbose': 1,
        'num_iterations': 800,
        'learning_rate': 0.3,
        'num_leaves': 500,
        'objective': 'binary'
    }
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        # 'booster': 'gblinear', # or gbtree
        # 'silent': 1, # do not log running messages
        'eta': 0.01,
        'max_depth': 15,
        'lambda': 1.5,
        # 'alpha': 0.4
    }

    dnn_params = {
        'output_size': 1,
        'epochs': 150,
        'batch_size': 32,
        'learning_rate': 0.001,
        'early_stopping_patience': 30
        }

    to_train = {
        "log_reg": False,
        'ridge': False,
        "knn": False,
        'tree': False,
        'random_forest': False,
        "catboost": False,
        'lgbm': False,
        'xgboost': False,
        "DNN": True
    }

    to_inference = {
        "log_reg": True,
        'ridge': True,
        "knn": False,
        'tree': False,
        'random_forest': True,
        "catboost": False,
        'lgbm': True,
        'xgboost': False,
        "DNN": True
    }