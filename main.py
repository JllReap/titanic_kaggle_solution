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


    def predict(test: pd.DataFrame, sample_sub: pd.DataFrame) -> pd.DataFrame:
        """Inference function."""

        df, data_checkpoint = dataset.get_dataset(test)
        preds = solver.predict(df, df_train, data_checkpoint)

        return sample_sub.with_columns(pd.Series('utility_agent1', preds))

if IS_TRAIN:
    artifacts = train(rerun=IS_RERUN, oof_features=None)

else:
    pass
