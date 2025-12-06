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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
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
            for model_name, flag in tqdm(config.to_inference.items()):
                if not flag:
                    continue
                self.models[model_name] = []

                ckpt_dir = config.path_to_load_solver_checkpoint[model_name]

                all_files = os.listdir(ckpt_dir)
                model_files = [
                    f for f in all_files
                    if f.startswith(f"main_{model_name}_") and f.endswith(".pickle")
                ]

                def _fold_num(fname: str) -> int:
                    return int(fname.split("_")[-1].split(".")[0])

                model_files = sorted(model_files, key=_fold_num)

                for fname in model_files:
                    full_path = os.path.join(ckpt_dir, fname)
                    with open(full_path, "rb") as f:
                        model = pickle.load(f)
                    self.models[model_name].append(model)

                print(f"{model_name}: loaded {len(self.models[model_name])} folds")

    def train_dnn(self, df, X, Y, model_name):
        # один раз делаем train/val
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X, Y, test_size=0.2, stratify=Y, random_state=42
        )

        base_model = DNN(**self.config.dnn_params)
        model = base_model
        model.fit(X_train, Y_train, X_valid, Y_valid)

        logits = model.predict(X_valid)  # np.array [N, 1]
        probs = torch.sigmoid(torch.tensor(logits)).numpy().ravel()
        preds = (probs > 0.5).astype(int)

        oof_preds = np.zeros(len(df))
        oof_labels = np.zeros(len(df))

        oof_preds[X_valid.index] = preds
        oof_labels[X_valid.index] = Y_valid.values

        score = accuracy_score(Y_valid, preds)
        print("DNN val accuracy:", round(score, 4))

        self.models[model_name]["models"] = [model]

        feature_importances = None
        return oof_labels, oof_preds, feature_importances

    def train_one_model(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        Y: pd.Series,
        cat_cols,
        model_name: str
    ) -> Tuple[np.ndarray, np.ndarray, Union[pd.DataFrame, None]]:
        """Train N folds of a certain model (including DNN)."""

        scores = []
        n = len(df)
        oof_preds = np.zeros(n)
        oof_labels = np.zeros(n)
        oof_mask = np.zeros(n)
        oof_baseline_preds = np.zeros(n)

        feature_importances = None
        oof_embeddings = None

        if self.config.use_baseline_scores:
            with open(self.config.path_to_load_solver_checkpoint["baseline"] + ".pickle", "rb") as f:
                baseline = pickle.load(f)["catboost"]["oof_baseline_preds"]
        else:
            baseline = None

        for fold in range(self.config.n_splits):

            print(f"\n{model_name} | Fold {fold}")

            train_index = df[df["fold"] != fold].index
            valid_index = df[df["fold"] == fold].index

            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            Y_train, Y_valid = Y.iloc[train_index], Y.iloc[valid_index]

            if self.config.use_baseline_scores and baseline is not None:
                baseline_train, baseline_valid = baseline[train_index], baseline[valid_index]
            else:
                baseline_train = baseline_valid = None

            if model_name == "log_reg":
                base_model = LogisticRegression(**self.config.log_reg_params)

            elif model_name == "ridge":
                base_model = RidgeClassifier(**self.config.ridge_params)

            elif model_name == "knn":
                base_model = KNeighborsClassifier(**self.config.knn_params)

            elif model_name == "tree":
                base_model = DecisionTreeClassifier(**self.config.tree_params)

            elif model_name == "random_forest":
                base_model = RandomForestClassifier(**self.config.random_forest_params)

            elif model_name == "catboost":
                if self.config.task == "classification":
                    base_model = CatBoostClassifier(
                        **self.config.catboost_params,
                        cat_features=cat_cols,
                    )
                else:
                    base_model = CatBoostRegressor(
                        **self.config.catboost_params,
                        cat_features=cat_cols,
                    )

            elif model_name == "lgbm":
                base_model = lgb.LGBMClassifier(**self.config.lgbm_params)

            elif model_name == "xgboost":
                base_model = xgb.XGBClassifier(**self.config.xgb_params)

            elif model_name == "DNN":
                base_model = DNN(**self.config.dnn_params)

            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            if not self.rerun:
                model = base_model

                if model_name == "catboost":
                    X_train_cat = X_train[cat_cols].astype(np.int64)
                    X_valid_cat = X_valid[cat_cols].astype(np.int64)

                    if self.config.use_baseline_scores and baseline_train is not None:
                        train_pool = Pool(
                            X_train_cat,
                            Y_train,
                            baseline=baseline_train,
                            cat_features=cat_cols,
                        )
                        valid_pool = Pool(
                            X_valid_cat,
                            Y_valid,
                            baseline=baseline_valid,
                            cat_features=cat_cols,
                        )
                    else:
                        train_pool = Pool(X_train_cat, Y_train, cat_features=cat_cols)
                        valid_pool = Pool(X_valid_cat, Y_valid, cat_features=cat_cols)

                    model.fit(train_pool, eval_set=valid_pool)

                elif model_name == "lgbm":
                    model.fit(
                        X_train,
                        Y_train,
                        eval_set=[(X_valid, Y_valid)],
                        eval_metric="rmse",
                        callbacks=[
                            lgb.early_stopping(self.config.lgbm_params["early_stopping_rounds"]),
                            lgb.log_evaluation(self.config.lgbm_params["verbose"]),
                        ],
                    )

                elif model_name == "xgboost":
                    model.fit(
                        X_train,
                        Y_train,
                        eval_set=[(X_valid, Y_valid)],
                        verbose=1000,
                    )

                elif model_name == "DNN":
                    model.fit(X_train, Y_train, X_valid, Y_valid)

                else:
                    model.fit(X_train, Y_train)

                os.makedirs(self.config.path_to_save_solver_checkpoint, exist_ok=True)
                path_to_save = os.path.join(
                    self.config.path_to_save_solver_checkpoint,
                    f"main_{model_name}_{fold}.pickle",
                )
                with open(path_to_save, "wb") as f:
                    pickle.dump(model, f)

            else:
                path_to_load = os.path.join(
                    self.config.path_to_load_solver_checkpoint[model_name],
                    f"main_{model_name}_{fold}.pickle",
                )
                with open(path_to_load, "rb") as f:
                    model = pickle.load(f)

            if model_name == "catboost":
                if self.config.task == "classification":
                    Y_valid = Y_valid.astype(np.float64)
                    preds = model.predict(X_valid[cat_cols].astype(np.int64)).astype(float)
                else:
                    if self.config.use_baseline_scores and baseline_valid is not None:
                        preds = model.predict(
                            Pool(
                                X_valid[cat_cols].astype(np.int64),
                                baseline=baseline_valid,
                                cat_features=cat_cols,
                            )
                        )
                    else:
                        preds = model.predict(
                            Pool(
                                X_valid[cat_cols].astype(np.int64),
                                cat_features=cat_cols,
                            )
                        )

            elif model_name == "lgbm":
                preds = model.predict(X_valid)

            elif model_name == "xgboost":
                preds = model.predict(X_valid)

            elif model_name == "DNN":
                logits, embeds = model.predict(X_valid, return_embeddings=True)

                # embeds: (len(valid_index), emb_dim)
                embeddings_df = pd.DataFrame(
                    embeds,
                    index=valid_index,
                    columns=[f"emb_{i}" for i in range(embeds.shape[1])]
                )
                embeddings_df["index"] = valid_index

                if oof_embeddings is None:
                    oof_embeddings = embeddings_df
                else:
                    oof_embeddings = pd.concat([oof_embeddings, embeddings_df], axis=0)

                print("DNN embeddings shape |", embeds.shape)

                logits = torch.tensor(logits).view(-1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().cpu().numpy()

            else:
                preds = model.predict(X_valid)

            # --- OOF ---
            oof_preds[valid_index] = preds
            oof_labels[valid_index] = Y_valid.values
            score = accuracy_score(Y_valid, preds)
            scores.append(score)

            print(f"Fold {fold} accuracy: {round(score, 4)}")

            if model_name == "catboost":
                fold_importance = model.get_feature_importance() / self.config.n_splits
                if feature_importances is None:
                    feature_importances = fold_importance
                else:
                    feature_importances += fold_importance

                fi = pd.DataFrame(
                    {
                        "feature": X_train[cat_cols].columns,
                        "importance": feature_importances,
                    }
                ).sort_values(by="importance", ascending=False)
                print(fi.head(10))

                if not os.path.exists("dataset"):
                    os.mkdir("dataset")
                fi.to_csv("dataset/feature_importance.csv", index=False)

            gc.collect()

        oof_preds = np.clip(oof_preds, -1, 1)
        oof_score = accuracy_score(oof_labels, oof_preds)

        self.models[model_name]["oof_baseline_preds"] = oof_baseline_preds
        self.models[model_name]["oof_preds"] = oof_preds
        self.models[model_name]["oof_labels"] = oof_labels
        self.models[model_name]["mode"] = oof_mask
        self.models[model_name]["fold"] = df["fold"].values

        print(f"\nCV scores {model_name}")
        for fold, s in enumerate(scores):
            print(f"Fold {fold} | {round(s, 4)}")
        print("AVG", round(np.mean(scores), 4))
        print("STD", round(np.std(scores), 4))
        print("OOF", round(oof_score, 4))

        if model_name == "catboost" and feature_importances is not None:
            feature_importances_df = pd.DataFrame(
                {
                    "feature": X_train[cat_cols].columns,
                    "importance": feature_importances,
                }
            ).sort_values(by="importance", ascending=False)
            print(feature_importances_df.head(10))
            feature_importances_df.to_csv("dataset/feature_importance.csv", index=False)
            feature_importances_ret = feature_importances_df
        else:
            feature_importances_ret = None

        if model_name == "DNN" and oof_embeddings is not None:
            oof_embeddings.sort_values(by="index").drop(["index"], axis=1).to_csv(
                "checkpoints/oof_cat_embeddings.csv",
                index=False,
            )

        return oof_labels, oof_preds, feature_importances_ret

    def fit(self, df: pd.DataFrame, data_checkpoint: dict):
        """Training function."""

        os.makedirs(self.config.path_to_save_solver_checkpoint, exist_ok=True)

        # Select features and target из уже подготовленного df
        X = df.drop(["Survived", 'fold'], axis=1)
        y = df["Survived"]

        cat_cols = data_checkpoint.get("cat_cols", [])

        artifacts = {}

        for model_name, to_train in self.config.to_train.items():
            if not to_train:
                continue

            print(f"\n==== Training {model_name} ====")
            self.models[model_name] = {}
            artifacts[model_name] = {}

            oof_labels, oof_preds, feature_importance = self.train_one_model(
                df, X, y, cat_cols, model_name
            )

            artifacts[model_name]["oof_preds"] = oof_preds
            artifacts[model_name]["oof_labels"] = oof_labels
            artifacts[model_name]["feature_importance"] = feature_importance

        # Save solution checkpoint for the inference.
        if self.config.is_train and not self.rerun:
            with open(self.config.path_to_save_solver_checkpoint + ".pickle", "wb") as f:
                pickle.dump(self.models, f)

        return artifacts


    def predict(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Run inference with all loaded models and compute metrics."""

        results = []

        for model_name, model_list in self.models.items():
            for i, model in enumerate(model_list):

                # Predict
                preds = model.predict(X)
                # Some models output probabilities → convert to labels
                if model_name == 'DNN':
                    logits = torch.tensor(preds).view(-1)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long().cpu().numpy()

                # Metrics
                acc = accuracy_score(y, preds)

                results.append({
                    "model": model_name,
                    "model_id": i,
                    "accuracy": acc,
                })

        return pd.DataFrame(results)