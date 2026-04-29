import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DETAIL_CSV = BASE_DIR / "frequency_tables" / "pump_formula_drug_dose_detail.csv"
PLOT_DIR = BASE_DIR / "plots"


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DETAIL_CSV)
    if df.empty:
        raise RuntimeError(f"No data in {DETAIL_CSV}")

    pivot = (
        df.pivot_table(index="drug_en", columns="dose_mg", values="count", aggfunc="sum", fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    x_labels = [f"{float(x):g}" for x in pivot.columns.to_list()]
    y_labels = pivot.index.to_list()
    count_matrix = pivot.to_numpy(dtype=float)

    # 1) Count heatmap
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(x_labels)), max(4.5, 0.6 * len(y_labels))))
    im = ax.imshow(count_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Dose (mg)")
    ax.set_ylabel("Drug")
    ax.set_title("Pump Formula: Drug-Dose Count Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pump_formula_dose_heatmap_count.png", dpi=200)
    plt.close(fig)

    # 2) Ratio heatmap (within each drug)
    row_sum = count_matrix.sum(axis=1, keepdims=True)
    ratio_matrix = np.divide(count_matrix, row_sum, out=np.zeros_like(count_matrix), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(x_labels)), max(4.5, 0.6 * len(y_labels))))
    im = ax.imshow(ratio_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Dose (mg)")
    ax.set_ylabel("Drug")
    ax.set_title("Pump Formula: Drug-Dose Ratio Heatmap (Within Drug)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Ratio")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pump_formula_dose_heatmap_ratio.png", dpi=200)
    plt.close(fig)

    print("Generated:")
    print(PLOT_DIR / "pump_formula_dose_heatmap_count.png")
    print(PLOT_DIR / "pump_formula_dose_heatmap_ratio.png")


if __name__ == "__main__":
    main()
