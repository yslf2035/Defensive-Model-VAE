import math
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def build_csv_path(data_dir: Path, model_name: str, scenario_name: str, trial_name: str) -> Path:
    """Build CSV path from model and scenario naming convention."""
    return data_dir / f"{model_name}_{scenario_name}_{trial_name}.csv"


def load_ego_speed_curve(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV and return DataFrame with normalized time and ego speed.

    Required columns:
    - sim_time
    - ego_vx
    - ego_vy
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"sim_time", "ego_vx", "ego_vy"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {sorted(missing_columns)}"
        )

    time = df["sim_time"].astype(float)
    time = time - time.iloc[0]  # Set first frame time to 0s.
    speed = (df["ego_vx"].astype(float) ** 2 + df["ego_vy"].astype(float) ** 2).apply(
        math.sqrt
    )

    return pd.DataFrame({"time_s": time, "ego_speed_mps": speed})


def plot_speed_time_curves(
    data_dir: Path,
    scenario_name: str,
    model_names: List[str],
    trial_name: str,
    figure_size: Tuple[float, float],
    line_width: float,
    show_window: bool,
    save_figure: bool,
    output_path: Path,
) -> None:
    """Plot ego speed-time curves for selected models and one scenario."""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=figure_size)

    plotted_any = False
    for model_name in model_names:
        csv_path = build_csv_path(data_dir, model_name, scenario_name, trial_name)
        if not csv_path.exists():
            print(f"[Warning] Skip missing file: {csv_path.name}")
            continue

        curve_df = load_ego_speed_curve(csv_path)
        plt.plot(
            curve_df["time_s"],
            curve_df["ego_speed_mps"],
            label=model_name,
            linewidth=line_width,
        )
        plotted_any = True

    if not plotted_any:
        raise FileNotFoundError(
            "No valid CSV files found. Check scenario/model settings in main()."
        )

    plt.title(f"Ego Speed vs Time - {scenario_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Ego Speed (m/s)")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    if save_figure:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=200)
        print(f"[Info] Figure saved to: {output_path}")
    if show_window:
        plt.show()
    else:
        plt.close()


def main() -> None:
    # 绘制SUT被测速度曲线
    # ------------------------ Editable Parameters -------------------------
    # Scenario can be changed to any scenario name in collected_data.
    # "StaticBlindTown05", "DynamicBlindTown05", "PredictableMovementTown05", "UnpredictableMovementTown04"
    scenario_name = "UnpredictableMovementTown04"

    # You can choose one or multiple model names.
    # Example: ["IDM"] or ["IDM", "BEHAVIOR", "TCP"]
    model_names = ["TCP"]

    # Trials: "1", "2", "3"
    trial_name = "2"

    # Plot style
    figure_size = (10, 6)
    line_width = 1.8

    # Display behavior: if Tk backend crashes in remote sessions, set show_window=False.
    show_window = False
    save_figure = True
    output_path = Path("results") / f"{'_'.join(model_names)}_{scenario_name}.png"
    # --------------------------------------------------------------------

    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / "collected_data"
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    # Disable Tk toolbar to avoid zero-size icon crash in some GUI environments.
    matplotlib.rcParams["toolbar"] = "None"

    plot_speed_time_curves(
        data_dir=data_dir,
        scenario_name=scenario_name,
        model_names=model_names,
        trial_name=trial_name,
        figure_size=figure_size,
        line_width=line_width,
        show_window=show_window,
        save_figure=save_figure,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
