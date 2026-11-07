import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

TARGETS = ["Green", "Dead", "Total"]

TRAIN_CSV = pathlib.Path(__file__).resolve().parent.parent / "data/train/train_wide.csv"
TOTAL_CSV = pathlib.Path(__file__).resolve().parent.parent / "data/raw/total.csv"
GREEN_COL = "materia verde seca por m2"
DEAD_COL = "materia muerta seca por m2"


def training_means() -> dict:
    df = pd.read_csv(TRAIN_CSV)
    green = float(df[GREEN_COL].mean())
    dead = float(df[DEAD_COL].mean())
    return {"Green": green, "Dead": dead, "Total": green + dead}


def dataset_means() -> dict:
    df = pd.read_csv(TOTAL_CSV)
    green = float(df[GREEN_COL].mean())
    dead = float(df[DEAD_COL].mean())
    return {"Green": green, "Dead": dead, "Total": green + dead}


def load_metrics_csv(log_path: pathlib.Path) -> pd.DataFrame:
    csv_path = pathlib.Path(log_path) / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró metrics.csv en {csv_path}")
    return pd.read_csv(csv_path)


def find_best_epoch(df: pd.DataFrame) -> int:
    return int(df.loc[df["BR2"].idxmax(), "Epoch"])


def _best_row(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["BR2"].idxmax()]


def _mark_best(ax: plt.Axes, df: pd.DataFrame) -> None:
    best_epoch = find_best_epoch(df)
    ax.axvline(best_epoch, color="black", linestyle="--", alpha=0.5, linewidth=1)


TARGET_COLORS = {"Green": "tab:green", "Dead": "tab:red", "Total": "tab:blue"}


def split_metric_runs(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Separa las corridas concatenadas en un metrics.csv (epoch que reinicia en 1)."""
    restarts = (df["Epoch"].diff() < 0).cumsum()
    return [group for _, group in df.groupby(restarts)]


def best_run_index(df: pd.DataFrame) -> int:
    runs = split_metric_runs(df)
    return max(range(len(runs)), key=lambda i: runs[i]["BR2"].max())


def select_best_run(df: pd.DataFrame) -> pd.DataFrame:
    return split_metric_runs(df)[best_run_index(df)]


def plot_metrics_evolution(df: pd.DataFrame, ax: plt.Axes) -> None:
    ax_r2 = ax
    ax_mae = ax.twinx()

    for target in TARGETS:
        ax_r2.plot(df["Epoch"], df[f"R2_{target}"], label=f"R² {target}",
                   color=TARGET_COLORS[target])
        ax_mae.plot(df["Epoch"], df[f"MAE_{target}"], label=f"MAE {target}",
                    color=TARGET_COLORS[target], linestyle="--")

    ax_r2.axhline(0.0, color="black", linewidth=0.6, alpha=0.25)
    ax_r2.set_ylim(0, 1)
    ax_r2.set_xlabel("Epoch")
    ax_r2.set_ylabel("R²")
    ax_mae.set_ylim(10, 150)
    ax_mae.set_ylabel("MAE")
    ax_r2.set_title("Evolución de R² (izq.) y MAE (der.) por target")
    ax_r2.grid(True, alpha=0.3)

    handles1, labels1 = ax_r2.get_legend_handles_labels()
    handles2, labels2 = ax_mae.get_legend_handles_labels()
    ax_mae.legend(handles1 + handles2, labels1 + labels2, loc="center right")
    _mark_best(ax_mae, df)


def make_validation_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    plot_metrics_evolution(df, ax)
    means = dataset_means()
    fig.suptitle(
        f"Métricas de validación (mejor modelo: época {find_best_epoch(df)} | "
        f"BR2: {_best_row(df)['BR2']:.4f})\n"
        f"Medias del dataset: Green {means['Green']:.1f} | "
        f"Dead {means['Dead']:.1f} | Total {means['Total']:.1f} {UNIT_LABEL}",
        fontsize=13,
    )
    return fig


def plot_train_loss(df: pd.DataFrame, ax: plt.Axes) -> None:
    ax.plot(df["Epoch"], df["Train_Loss"], label="Train Loss", color="tab:red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    _mark_best(ax, df)


def make_optimization_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    plot_train_loss(df, ax)
    return fig


def build_training_figures(df: pd.DataFrame) -> list[plt.Figure]:
    df = select_best_run(df)
    return [make_validation_figure(df), make_optimization_figure(df)]


def show_training_plots(df: pd.DataFrame, block: bool = True) -> None:
    build_training_figures(df)
    plt.show(block=block)


# --- Tema de evaluación (estilo publicación científica) ---
PRED_GREEN_COL = "Dry_Green_g"
PRED_DEAD_COL = "Dry_Dead_g"

TRAIN_COLOR = "#2E5E8C"   # azul acero
TEST_COLOR = "#B0413E"    # rojo ladrillo
GREEN_TARGET_COLOR = "#3A7D44"
DEAD_TARGET_COLOR = "#9E4A2E"

UNIT_LABEL = r"g m$^{-2}$"
GREEN_LABEL = f"Materia verde seca ({UNIT_LABEL})"
DEAD_LABEL = f"Materia muerta seca ({UNIT_LABEL})"
OBSERVED_LABEL = f"Valor observado ({UNIT_LABEL})"
PREDICTED_LABEL = f"Valor predicho ({UNIT_LABEL})"

EVAL_RC_PARAMS = {
    # Tipografía serif, consistente en toda la suite
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titlepad": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    # Fondo blanco y líneas finas
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    # Caja completa con ticks hacia adentro (estilo journal)
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    # Leyendas sin marco en todas las figuras
    "legend.frameon": False,
}


def apply_evaluation_style() -> None:
    plt.rcParams.update(EVAL_RC_PARAMS)


def format_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)


def regression_stats(real: np.ndarray, pred: np.ndarray) -> tuple[float, float, int]:
    """R², MAE y n para el panel Real vs Predicho."""
    ss_res = np.sum((real - pred) ** 2)
    ss_tot = np.sum((real - real.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(pred - real))
    return float(r2), float(mae), len(real)


def mean_logged_mae(version_logs_dir: pathlib.Path) -> dict[str, float] | None:
    """Promedia, entre folds de logs/<versión>, el MAE de la mejor época
    (BR2 máximo) de la mejor corrida de cada fold. None si no hay logs."""
    fold_dirs = sorted(
        p for p in version_logs_dir.glob("fold_*") if (p / "metrics.csv").exists()
    )
    best_rows = [ _best_row(select_best_run(load_metrics_csv(d))) for d in fold_dirs ]
    if not best_rows:
        return None
    return {
        "Green": float(np.mean([row["MAE_Green"] for row in best_rows])),
        "Dead": float(np.mean([row["MAE_Dead"] for row in best_rows])),
    }


def plot_mass_distribution(
    ax: plt.Axes,
    green_train: pd.Series,
    dead_train: pd.Series,
    green_test: pd.Series,
    dead_test: pd.Series,
) -> None:
    ax.scatter(
        green_train, dead_train, color=TRAIN_COLOR, label=f"Entrenamiento (n={len(green_train)})",
        s=45, alpha=0.85, edgecolors="white", linewidths=0.6,
    )
    ax.scatter(
        green_test, dead_test, color=TEST_COLOR, label=f"Test (n={len(green_test)})",
        s=45, alpha=0.85, edgecolors="white", linewidths=0.6,
    )
    ax.set_xlabel(GREEN_LABEL)
    ax.set_ylabel(DEAD_LABEL)
    ax.set_title("Distribución de masas observadas")
    ax.legend(loc="upper right")
    format_axes(ax)


def plot_histogram_comparison(
    ax: plt.Axes, train: pd.Series, test: pd.Series, bins: int = 15,
    title: str = "", xlabel: str = "",
) -> None:
    combined = pd.concat([train, test]).dropna()
    edges = np.histogram_bin_edges(combined, bins=bins)
    for data, color, name in ((train, TRAIN_COLOR, "Entrenamiento"),
                              (test, TEST_COLOR, "Test")):
        data = data.dropna()
        ax.hist(data, bins=edges, color=color, alpha=0.75,
                edgecolor="white", linewidth=0.6,
                label=f"{name} (n={len(data)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title(title)
    ax.legend(loc="upper right")
    format_axes(ax)


def plot_identity_scatter(
    ax: plt.Axes, real: pd.Series, pred: pd.Series,
    color: str, title: str, err_mae: float | None = None,
) -> None:
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)

    top = float(max(real.max(), pred.max()))
    if err_mae is not None:
        top += err_mae
    lim = top * 1.06

    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=0.9,
            alpha=0.7, zorder=1, label="1:1")
    if err_mae is not None:
        ax.errorbar(real, pred, yerr=err_mae, fmt="none",
                    ecolor=color, elinewidth=0.9, alpha=0.45, zorder=1.5,
                    label=r"$\pm$MAE (val)")
    ax.scatter(real, pred, color=color, s=45, alpha=0.85,
               edgecolors="black", linewidths=0.5, zorder=2)

    r2, mae, n = regression_stats(real, pred)
    mae_line = f"MAE $= {mae:.1f}$ {UNIT_LABEL} (test)"
    if err_mae is not None:
        mae_line += f"\nBarras $\\pm {err_mae:.1f}$ {UNIT_LABEL} (val)"
    stats_text = f"$R^2 = {r2:.3f}$\n{mae_line}\n$n = {n}$"
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=10)

    ax.set_xlabel(OBSERVED_LABEL)
    ax.set_ylabel(PREDICTED_LABEL)
    ax.set_title(title)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    format_axes(ax)


def make_mass_distribution_figure(data) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    plot_mass_distribution(
        ax, data.green_train, data.dead_train, data.green_test, data.dead_test
    )
    fig.tight_layout()
    return fig


def make_histogram_figure(train: pd.Series, test: pd.Series, bins: int = 15,
                          title: str = "", xlabel: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    plot_histogram_comparison(ax, train, test, bins=bins, title=title, xlabel=xlabel)
    fig.tight_layout()
    return fig


def make_identity_figure(data, mae_green: float | None = None,
                         mae_dead: float | None = None) -> plt.Figure:
    fig, (ax_green, ax_dead) = plt.subplots(1, 2, figsize=(11.0, 5.6))
    plot_identity_scatter(
        ax_green, data.merged[GREEN_COL], data.merged[PRED_GREEN_COL],
        color=GREEN_TARGET_COLOR, title="Materia verde seca",
        err_mae=mae_green,
    )
    plot_identity_scatter(
        ax_dead, data.merged[DEAD_COL], data.merged[PRED_DEAD_COL],
        color=DEAD_TARGET_COLOR, title="Materia muerta seca",
        err_mae=mae_dead,
    )
    for ax, letter in ((ax_green, "(a)"), (ax_dead, "(b)")):
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
                fontsize=12, fontweight="bold", ha="left", va="bottom")
    fig.tight_layout(pad=1.5)
    return fig


def build_evaluation_figures(data, bins: int = 15,
                             mae_green: float | None = None,
                             mae_dead: float | None = None) -> list[plt.Figure]:
    apply_evaluation_style()
    return [
        make_mass_distribution_figure(data),
        make_histogram_figure(data.green_train, data.green_test, bins=bins,
                              title="Distribución de materia verde seca",
                              xlabel=GREEN_LABEL),
        make_histogram_figure(data.dead_train, data.dead_test, bins=bins,
                              title="Distribución de materia muerta seca",
                              xlabel=DEAD_LABEL),
        make_identity_figure(data, mae_green=mae_green, mae_dead=mae_dead),
    ]


def show_evaluation_plots(data, bins: int = 15, block: bool = True,
                          mae_green: float | None = None,
                          mae_dead: float | None = None) -> None:
    build_evaluation_figures(data, bins=bins,
                             mae_green=mae_green, mae_dead=mae_dead)
    plt.show(block=block)