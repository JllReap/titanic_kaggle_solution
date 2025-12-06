import pandas as pd

from config import Config, IS_TRAIN, IS_RERUN
from dataset import Dataset
from solver import Solver
from utils import set_seed


# --- Train and inference ---

def train(rerun: bool, oof_features=None) -> dict:
    """Training function."""

    config = Config()

    set_seed(config.seed)
    dataset = Dataset(config, rerun)
    solver = Solver(config, rerun)

    df = pd.read_csv(config.path_to_train)
    df, data_checkpoint = dataset.get_dataset(df)
    artifacts = solver.fit(df, data_checkpoint)

    return artifacts


if not IS_TRAIN:
    config = Config()

    set_seed(config.seed)
    dataset = Dataset(config, rerun=False)
    solver = Solver(config, rerun=False)

    df_train = None  # Not used, set to None to save RAM.


    def predict() -> pd.DataFrame:
        config = Config()

        set_seed(config.seed)
        dataset = Dataset(config, rerun=False)
        solver = Solver(config, rerun=False)

        df_test_raw = pd.read_csv(config.path_to_test)
        df_labels = pd.read_csv("titanic/gender_submission.csv")

        df_merged = df_test_raw.merge(df_labels, on="PassengerId")
        y_test = df_merged["Survived"].astype(int)

        X_test_raw = df_merged.drop(columns=["Survived"])

        X_test, data_checkpoint = dataset.get_dataset(X_test_raw)

        return solver.predict(X_test, y_test)

if IS_TRAIN:
    artifacts = train(rerun=IS_RERUN, oof_features=None)

else:
    result = predict()
    print(result[result['model_id'] == 0].sort_values(by='accuracy', ascending=False))
