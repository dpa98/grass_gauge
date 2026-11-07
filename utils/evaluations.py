import pathlib
from dataclasses import dataclass

import pandas as pd

from .plotting import DEAD_COL, GREEN_COL, PRED_DEAD_COL, PRED_GREEN_COL

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
REALS_CSV = PROJECT_ROOT / "data" / "raw" / "total.csv"
PREDS_CSV = PROJECT_ROOT / "predictions.csv"

REAL_TARGET_COLS = [GREEN_COL, DEAD_COL]
PRED_TARGET_COLS = [PRED_GREEN_COL, PRED_DEAD_COL]
ID_CANDIDATES = ("id", "Id", "ID")


@dataclass
class EvaluationData:
    train: pd.DataFrame
    test: pd.DataFrame
    merged: pd.DataFrame

    @property
    def green_train(self) -> pd.Series:
        return self.train[GREEN_COL]

    @property
    def dead_train(self) -> pd.Series:
        return self.train[DEAD_COL]

    @property
    def green_test(self) -> pd.Series:
        return self.test[GREEN_COL]

    @property
    def dead_test(self) -> pd.Series:
        return self.test[DEAD_COL]


def find_id_column(df: pd.DataFrame, label: str) -> str:
    for candidate in ID_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"No se encontró columna de ID en {label}. Columnas disponibles: {list(df.columns)}"
    )


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing} en {label}. Columnas disponibles: {list(df.columns)}"
        )


def load_reals(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, REAL_TARGET_COLS, str(path))
    return df


def load_predictions(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, PRED_TARGET_COLS, str(path))
    return df


def prepare_evaluation_data(reals_csv: pathlib.Path, preds_csv: pathlib.Path) -> EvaluationData:
    """Carga reales y predicciones, separa Train/Test y empareja 1:1 por ID."""
    df_reals = load_reals(reals_csv)
    df_preds = load_predictions(preds_csv)

    id_reals = find_id_column(df_reals, str(reals_csv))
    id_preds = find_id_column(df_preds, str(preds_csv))

    test_ids = df_preds[id_preds].dropna().unique()
    test_mask = df_reals[id_reals].isin(test_ids)

    train = df_reals[~test_mask].dropna(subset=REAL_TARGET_COLS)
    test = df_reals[test_mask].dropna(subset=REAL_TARGET_COLS)

    preds_unique = df_preds.dropna(subset=[id_preds]).drop_duplicates(subset=[id_preds])
    merged = (
        df_reals.merge(preds_unique, left_on=id_reals, right_on=id_preds, how="inner")
        .dropna(subset=REAL_TARGET_COLS + PRED_TARGET_COLS)
    )

    if merged.empty:
        raise ValueError(
            "Ningún ID de las predicciones coincide con las muestras de campo "
            f"({len(test_ids)} IDs en {preds_csv.name})."
        )

    return EvaluationData(train=train, test=test, merged=merged)
