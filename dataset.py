import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from config import Config


# --- Data Pipeline ---

def get_age_cat(x: int):
    if x < 15:
        return "child"

    elif x >= 15 and x < 30:
        return 'young man'

    elif x >= 30 and x < 50:
        'adult'

    elif x >= 50:
        return 'old man'

def get_family_cat(x: int):
    if x == 1:
        return "alone"

    elif x > 1 and x <= 3:
        return "small"

    elif x > 3 and x < 6:
        return "medium"

    else:
        return "large"

class Dataset():
    def __init__(self, config: Config, rerun: bool) -> None:
        """Initialization."""
        self.config = config
        self.rerun = rerun

        self.preprocess: ColumnTransformer | None = None
        self.cols_to_drop: list[str] = ['PassengerId', 'Ticket', 'Name', "Cabin", 'Fare']

        if self.config.is_train:
            self.data_checkpoint = {}
        else:
            with open(config.path_to_load_data_checkpoint, "rb") as f:
                self.data_checkpoint = pickle.load(f)

    def save_data_checkpoint(self, df: pd.DataFrame) -> None:
        with open(self.config.path_to_load_data_checkpoint, "wb") as f:
            pickle.dump(self.data_checkpoint, f)

        df.to_csv(self.config.path_to_save_dataset, index=False)

    def find_outliers(self, df: pd.DataFrame, cols: list[str]) -> dict[str, tuple[float, float]]:
        bounds = {}
        cols = list(cols)

        for col in cols:
            s = df[col].dropna()
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr

            bounds[col] = (low, high)

        return bounds

    def fit(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        num_scaler = StandardScaler() if self.config.use_std_scaler else MinMaxScaler()
        cat_encoder = OneHotEncoder(handle_unknown='ignore')
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        num_transformer = Pipeline(steps=[
            ('imputer', num_imputer),
            ('scaler', num_scaler)
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', cat_imputer),
            ('encoder', cat_encoder)
        ])

        self.preprocess = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ],
            remainder='drop',
            verbose_feature_names_out="{transformer_name}_{feature_name}"
        )

        self.preprocess.fit(X)
        self.outliers_borders = self.find_outliers(X, num_cols)

        self.data_checkpoint['num_cols'] = num_cols
        self.data_checkpoint['cat_cols'] = cat_cols
        self.data_checkpoint["preprocess"] = self.preprocess
        self.data_checkpoint['outliers_borders'] = self.outliers_borders
        self.data_checkpoint["cols_to_drop"] = self.cols_to_drop

    def clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip values. Don't drop them, but change to border value"""
        df = df.copy()
        for col, (low, high) in self.data_checkpoint['outliers_borders'].items():
            if col not in df.columns:
                continue
            df[col] = df[col].clip(lower=low, upper=high)
        return df

    def preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing function"""
        # scaling and encoding
        X = self.clip_outliers(X)
        X_arr = self.data_checkpoint['preprocess'].transform(X)
        new_features_names = self.data_checkpoint['preprocess'].get_feature_names_out()
        self.data_checkpoint['cat_cols'] = new_features_names
        X_df = pd.DataFrame(data=X_arr, columns=new_features_names, index=X.index)
        return X_df


    def drop_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Function to drop empty/unnecessary columns"""
        return df.drop(columns=[c for c in self.cols_to_drop if c in df.columns], axis=1, errors='ignore')

    def feature_engineering(self, X):
        """Function to create new features"""
        X.loc[:, 'Deck'] = X.loc[:, 'Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
        X.loc[:, 'Deck'] = X['Deck'].replace(['A', 'B', 'C'], 'ABC')
        X.loc[:, 'Deck'] = X.loc[:, 'Deck'].replace(['D', 'E'], 'DE')
        X.loc[:, 'Deck'] = X.loc[:, 'Deck'].replace(['F', 'G'], 'FG')
        X.loc[:, 'Family_Size'] = X.loc[:, 'SibSp'] + X.loc[:, 'Parch'] + 1
        X.loc[:, "Family_Category"] = X.loc[:, 'Family_Size'].apply(get_family_cat)
        X.loc[:, 'Ticket_Frequency'] = X.groupby('Ticket')['Ticket'].transform('count')

        return X

    def build_validation_and_cv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the validation and CV features."""

        if self.config.is_train:

            # Build the validation based on the original data

            src_df = df.copy()

            y = src_df['Survived'].values
            skf = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=True,
                random_state=self.config.seed,
            )
            for fold, (_, val_idx) in enumerate(skf.split(df, y)):
                src_df.loc[val_idx, "fold"] = fold

            return src_df

        else:
            return df

    def reduce_mem_usage(self, df, float16_as32=True):
        """Reduce memore usage."""

        start_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object and str(col_type) != 'category':
                c_min, c_max = df[col].min(), df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        if float16_as32:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df

    def get_dataset(self, df: pd.DataFrame):
        if self.config.is_train and self.config.load_dataset_checkpoint:
            print("Loading the dataset from the checkpoint...")

            df = pd.read_csv(self.config.path_to_save_dataset)

            with open(self.config.path_to_load_data_checkpoint, "rb") as f:
                self.data_checkpoint = pickle.load(f)

            cat_cols = self.data_checkpoint["cat_cols"]
            cat_mapping = {f: "str" if f in cat_cols else float for f in df.columns}
            df = df.astype(cat_mapping)
            print(df.head())

            return df, self.data_checkpoint

        if self.config.is_train:
            y = df['Survived']
            X = df.drop(["Survived"], axis=1)

            X = X.astype({"Pclass": "object"})
            X = self.feature_engineering(X)
            X = self.drop_cols(X)

            # preprocessing
            self.fit(X)
            X_proc = self.preprocessing(X)
            df_proc = pd.concat([X_proc, y], axis=1)

            # folds
            df_proc = self.build_validation_and_cv_features(df_proc)

            # optimise and save
            df_proc = self.reduce_mem_usage(df_proc)
            self.save_data_checkpoint(df_proc)

            print(df_proc.head())

            return df_proc, self.data_checkpoint

        X = df.copy()
        X = X.astype({"Pclass": "object"})
        X = self.feature_engineering(X)
        X = self.drop_cols(X)

        # make preprocessing with checkpoint
        X_proc = self.preprocessing(X)
        X_proc = self.build_validation_and_cv_features(X_proc)
        X_proc = self.reduce_mem_usage(X_proc)

        return X_proc, self.data_checkpoint