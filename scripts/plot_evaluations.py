import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

try:
    matplotlib.use("TkAgg")  # Previene el crash del servidor gráfico en i3/Linux
except ImportError:
    pass

from config import Config
from utils.evaluations import PREDS_CSV, REALS_CSV, prepare_evaluation_data
from utils.plotting import (
    DEAD_COL,
    GREEN_COL,
    PRED_DEAD_COL,
    PRED_GREEN_COL,
    UNIT_LABEL,
    mean_logged_mae,
    regression_stats,
    show_evaluation_plots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera la suite de gráficos de evaluación del modelo (Train vs Test)."
    )
    parser.add_argument("--reals", type=pathlib.Path, default=REALS_CSV,
                        help="CSV con todas las muestras de campo")
    parser.add_argument("--preds", type=pathlib.Path, default=PREDS_CSV,
                        help="CSV con las predicciones (Test)")
    parser.add_argument("--bins", type=int, default=15,
                        help="Cantidad de bins para los histogramas (default: 15)")
    parser.add_argument("--version", type=str, default="v1",
                        help="Versión del experimento cuyos logs proveen el MAE "
                             "de las barras de error (default: v1)")
    parser.add_argument("--no-block", action="store_true",
                        help="No bloquear la ejecución al mostrar las figuras")
    return parser.parse_args()


def _test_mae(real, pred) -> float:
    return regression_stats(real.to_numpy(dtype=float),
                            pred.to_numpy(dtype=float))[1]


def resolve_error_maes(args, data):
    """MAE por panel para las barras: promedio de logs o fallback a test."""
    logged = mean_logged_mae(Config(version=args.version).logs_dir / args.version)
    if logged is not None:
        print(f"[*] Barras con MAE promedio de logs ({args.version}): "
              f"Green {logged['Green']:.1f} | Dead {logged['Dead']:.1f} {UNIT_LABEL}")
        return logged["Green"], logged["Dead"]

    mae_green = _test_mae(data.merged[GREEN_COL], data.merged[PRED_GREEN_COL])
    mae_dead = _test_mae(data.merged[DEAD_COL], data.merged[PRED_DEAD_COL])
    print(f"[!] Sin logs para '{args.version}': barras con MAE de test "
          f"(Green {mae_green:.1f} | Dead {mae_dead:.1f} {UNIT_LABEL})")
    return mae_green, mae_dead


def main():
    args = parse_args()

    for path in (args.reals, args.preds):
        if not path.exists():
            print(f"Error: No se encontró el archivo: {path}")
            sys.exit(1)

    print(f"[*] Cargando datos reales desde: {args.reals}")
    print(f"[*] Cargando predicciones desde: {args.preds}")

    try:
        data = prepare_evaluation_data(args.reals, args.preds)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"[*] Train: {len(data.train)} | Test: {len(data.test)} | "
          f"Predicciones emparejadas: {len(data.merged)}")

    mae_green, mae_dead = resolve_error_maes(args, data)

    print("[*] Mostrando gráficos en pantalla.")
    show_evaluation_plots(data, bins=args.bins, block=not args.no_block,
                          mae_green=mae_green, mae_dead=mae_dead)


if __name__ == "__main__":
    main()
