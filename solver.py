import os
import pickle
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from config import Config
from dnn import DNN

class Solver:
    """Solution."""

    def __init__(self, config: Config, rerun: bool) -> None:
        """Initialiaztion."""

        self.config = config
        self.rerun = rerun
        self.models = {}

        if not config.is_train:
            print("Loading checkpoints for inference.")
            for checkpoint in tqdm(config.to_inference):
                self.models[checkpoint] = []
                model_paths = os.listdir(config.path_to_load_solver_checkpoint[checkpoint])
                for path in model_paths:
                    with open(os.path.join(config.path_to_load_solver_checkpoint[checkpoint], path), 'rb') as file:
                        model = pickle.load(file)
                        self.models[checkpoint].append(model)

    def train_dnn(self, df, X, Y, model_name):
        # один раз делаем train/val
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X, Y, test_size=0.2, stratify=Y, random_state=42
        )

        base_model = DNN(**self.config.dnn_params)
        model = base_model
        model.fit(X_train, Y_train, X_valid, Y_valid)

        # предикты на валидации
        logits = model.predict(X_valid)  # np.array [N, 1] логиты
        probs = torch.sigmoid(torch.tensor(logits)).numpy().ravel()
        preds = (probs > 0.5).astype(int)

        oof_preds = np.zeros(len(df))
        oof_labels = np.zeros(len(df))

        # здесь мы честного OOF не имеем, поэтому:
        oof_preds[X_valid.index] = preds
        oof_labels[X_valid.index] = Y_valid.values

        score = accuracy_score(Y_valid, preds)
        print("DNN val accuracy:", round(score, 4))

        # сохраним модель
        self.models[model_name]["models"] = [model]

        feature_importances = None  # для DNN их нет
        return oof_labels, oof_preds, feature_importances

    def train_one_model(self, df, X, Y, cat_cols, model_name) -> Tuple[np.array, np.array, Union[pd.DataFrame, None]]:
        """Train N folds of a certain model."""

        # Define the instances for the metrics.

        scores = []
        oof_preds, oof_labels, oof_mask = np.zeros([len(df)]), np.zeros([len(df)]), np.zeros([len(df)])
        oof_baseline_preds = np.zeros([len(df)])

        if self.config.use_baseline_scores:
            with open(self.config.path_to_load_solver_checkpoint["baseline"] + '.pickle', "rb") as f:
                baseline = pickle.load(f)["catboost"]["oof_baseline_preds"]

        if model_name == "catboost":
            feature_importances = None

        oof_embeddings = None

        for fold in range(self.config.n_splits):

            # Log model training
            print(f"\n{model_name} | Fold {fold}")

            train_index = df[df["fold"] != fold].index
            valid_index = df[df["fold"] == fold].index

            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            Y_train, Y_valid = Y.iloc[train_index], Y.iloc[valid_index]

            if self.config.use_baseline_scores:
                baseline_train, baseline_valid = baseline[train_index], baseline[valid_index]

            # print("X-train and X-valid shapes", X_train.shape, X_valid.shape)
            if model_name == "log_reg":
                base_model = LogisticRegression(**self.config.log_reg_params)

            elif model_name == "ridge":
                print(self.config.ridge_params)
                base_model = RidgeClassifier(**self.config.ridge_params)

            elif model_name == 'knn':
                base_model = KNeighborsClassifier(**self.config.knn_params)

            elif model_name == 'tree':
                base_model = DecisionTreeClassifier(**self.config.tree_params)

            elif model_name == "random_forest":
                base_model = RandomForestClassifier(**self.config.random_forest_params)

            elif model_name == "catboost":
                if self.config.task == "classification":
                    base_model = CatBoostClassifier(**self.config.catboost_params, cat_features=cat_cols)
                else:
                    base_model = CatBoostRegressor(**self.config.catboost_params, cat_features=cat_cols)

            elif model_name == "lgbm":
                base_model = lgb.LGBMClassifier(**self.config.lgbm_params)

            elif model_name == "xgboost":
                base_model = xgb.XGBClassifier(**self.config.xgb_params)

            elif model_name == "DNN":
                base_model = DNN(**self.config.dnn_params)

            if not self.rerun:

                # print("\nMain model")

                model = base_model

                if model_name == "catboost":
                    X_train = X_train[cat_cols].astype(np.int64)
                    X_valid = X_valid[cat_cols].astype(np.int64)

                    if self.config.use_baseline_scores:
                        train_pool = Pool(X_train, Y_train, baseline=baseline_train, cat_features=cat_cols)
                        valid_pool = Pool(X_valid, Y_valid, baseline=baseline_valid, cat_features=cat_cols)
                    else:
                        train_pool = Pool(X_train, Y_train, cat_features=cat_cols)
                        valid_pool = Pool(X_valid, Y_valid, cat_features=cat_cols)
                    model.fit(train_pool, eval_set=valid_pool)

                elif model_name == "lgbm":
                    model.fit(X_train, Y_train,
                              eval_set=[(X_valid, Y_valid)],
                              eval_metric='rmse',
                              callbacks=[
                                  lgb.early_stopping(self.config.lgbm_params["early_stopping_rounds"]),
                                  lgb.log_evaluation(self.config.lgbm_params["verbose"])
                              ])

                elif model_name == "xgboost":
                    model.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)], verbose=1000)

                elif model_name == "DNN":
                    model.fit(X_train, Y_train, X_valid, Y_valid)

                else:
                    model.fit(X_train, Y_train)

            else:
                path_to_load = os.path.join(self.config.path_to_load_solver_checkpoint[model_name],
                                            f"main_{model_name}_{fold}.pickle")
                with open(path_to_load, "rb") as f:
                    model = pickle.load(f)

            if model_name == "catboost":
                if self.config.task == "classification":
                    Y_valid = Y_valid.astype(np.float64)
                    preds = model.predict(X_valid).astype(float)
                else:
                    if self.config.use_baseline_scores:
                        preds = model.predict(Pool(X_valid, baseline=baseline_valid, cat_features=cat_cols))
                    else:
                        preds = model.predict(Pool(X_valid, cat_features=cat_cols))

            elif model_name == "lgbm":
                preds = model.predict(X_valid)

            elif model_name == "xgboost":
                preds = model.predict(X_valid)

            elif model_name == "DNN":
                preds = model.predict(X_valid)
                preds = torch.tensor(preds).sigmoid()
                preds = (preds > 0.5).long().cpu().numpy()
            else:
                preds = model.predict(X_valid)

            # Save the scores and the metrics.

            oof_preds[valid_index] = preds
            oof_labels[valid_index] = Y_valid

            score = accuracy_score(Y_valid, preds)
            scores.append(score)

            print(round(score, 4))

            # SHAP.
            # if self.config.show_shap:
            #     explainer = shap.TreeExplainer(model)
            #     shap_values = explainer(X_valid_src[:1000])
            #     shap.plots.beeswarm(shap_values, max_display=20)

            if not self.rerun:
                path_to_save = os.path.join(self.config.path_to_save_solver_checkpoint, f"main_{model_name}_{fold}.pickle")
                with open(path_to_save, "wb") as f:
                    pickle.dump(model, f)

            if model_name == "catboost":
                if feature_importances is None:
                    feature_importances = model.get_feature_importance() / self.config.n_splits
                else:
                    feature_importances += model.get_feature_importance() / self.config.n_splits
                fi = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": feature_importances
                }).sort_values(by='importance', ascending=False)
                print(fi.head(10))
                if not os.path.exists("dataset"):
                    os.mkdir("dataset")

                fi.to_csv('dataset/feature_importance.csv')

            gc.collect()

        # Clip final predictions.

        oof_preds = np.clip(oof_preds, -1, 1)
        oof_score = accuracy_score(oof_labels, oof_preds)

        self.models[model_name]["oof_baseline_preds"] = oof_baseline_preds
        self.models[model_name]["oof_preds"] = oof_preds
        self.models[model_name]["oof_labels"] = oof_labels
        self.models[model_name]["mode"] = oof_mask
        self.models[model_name]["fold"] = df["fold"].values

        print(f'\nCV scores {model_name}')
        for fold in range(len(scores)):
            print(f'Fold {fold} | {round(scores[fold], 4)}')

        print("AVG", round(np.mean(scores), 4))
        print("STD", round(np.std(scores), 4))
        print("OOF", round(oof_score, 4))

        if model_name == "catboost":
            feature_importances = pd.DataFrame({
                "feature": X_train.columns,
                "importance": feature_importances
            }).sort_values(by='importance', ascending=False)
            print(feature_importances.head(10))
            feature_importances.to_csv('dataset/feature_importance.csv')
        else:
            feature_importances = None

        if model_name == "DNN" and oof_embeddings is not None:
            oof_embeddings.sort_values(by="index").drop(["index"], axis=1).to_csv('checkpoints/oof_cat_embeddings.csv',
                                                                                  index=False)

        return oof_labels, oof_preds, feature_importances

    def fit(self, df: pd.DataFrame, data_checkpoint: dict):
        """Training function"""

        # Create checkpoint folder.
        if not os.path.exists(self.config.path_to_save_solver_checkpoint):
            os.mkdir(self.config.path_to_save_solver_caheckpoint)

        # Select features and target
        X = df.drop(["Survived"], axis = 1)
        y = df["Survived"]

        # Get categorical features
        cat_cols = data_checkpoint['cat_cols']

        artifacts = {}

        for model_name, to_train in self.config.to_train.items():
            if to_train:
                self.models[model_name] = {}
                artifacts[model_name] = {}

                if model_name != 'DNN':
                    df = pd.read_csv("titanic/train.csv")
                    X = df.drop(["Survived"], axis=1)
                    y = df["Survived"]
                    oof_labels, oof_preds, feature_importance = self.train_one_model(df, X, y, cat_cols, model_name)
                else:
                    oof_labels, oof_preds, feature_importance = self.train_dnn(df, X, y, model_name)

                artifacts[model_name]["oof_preds"] = oof_preds
                artifacts[model_name]["oof_labels"] = oof_labels
                artifacts[model_name]["feature_importance"] = feature_importance

        # Save solution checkpoint for the inference.
        if self.config.is_train and not self.rerun:
            with open(self.config.path_to_save_solver_checkpoint + '.pickle', "wb") as f:
                pickle.dump(self.models, f)
        elif self.rerun:
            pass

        return artifacts

    def predict(self, X: pd.DataFrame, df_train: pd.DataFrame, data_checkpoint: dict) -> np.array:
        """Inference."""

        # Process data.
        X = X.drop(["Survived"], axis=1)

        # Inference.
        # Order is important.
        # Weights from Ridge.

        prediction = np.zeros(len(X))

        # Catboost classic.

        model_name = "catboost_classic"

        preds = np.zeros(len(X))
        for model in self.models[model_name]:
            preds += model.predict(X) / len(self.models[model_name])

        prediction += np.clip(preds, -1, 1) # * 0.56566558 # weight: can be tuned

        # Catboost nested.

        model_name = "catboost_nested"

        preds = np.zeros(len(X))
        for model in self.models[model_name]:
            preds += model.predict(X) / len(self.models[model_name])

        prediction += np.clip(preds, -1, 1) # * 0
        X["oof"] = np.clip(preds, -1, 1)

        # Catboost OOF.

        model_name = "catboost_oof"

        preds = np.zeros(len(X))
        for model in self.models[model_name]:
            preds += model.predict(X) / len(self.models[model_name])

        prediction += np.clip(preds, -1, 1) # * 0

        # LGBM OOF.

        model_name = "lgbm_oof"

        preds = np.zeros(len(X))
        for model in self.models[model_name]:
            preds += model.predict(X) / len(self.models[model_name])

        prediction += np.clip(preds, -1, 1) # * 0.05761242
        # DNN OOF.

        model_name = "dnn_oof"

        preds = np.zeros(len(X))
        for model in self.models[model_name]:
            preds += model.predict(X) / len(self.models[model_name])

        prediction += np.clip(preds, -1, 1) # * 0.00826429

        # Catboost with baseline initialization.

        model_name = "catboost_with_oof_baseline"

        preds = np.zeros(len(X))
        for model in self.models[model_name]:
            preds += model.predict(
                Pool(
                    X.drop(["oof"], axis=1),
                    baseline=X["oof"],
                    cat_features=data_checkpoint["catcols"]
                )
            ) / len(self.models[model_name])

        prediction += np.clip(preds, -1, 1) # * 0.43124984

        return prediction