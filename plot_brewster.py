from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_brewster_dataset(path: Path) -> Tuple[str, np.ndarray]:
    """Load a Brewster-angle dataset, returning the label and numeric measurements."""
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        raw_title = handle.readline().strip()
    data = np.loadtxt(path, skiprows=2)
    label = raw_title.replace("Polarization of Light 'Brewsters' Data File", "").strip()
    return label, data


def plot_brewster_experiment(data: np.ndarray, label: str, output_path: Path) -> tuple[float, float, int]:
    """Render Brewster-angle diagnostics and return MSE, chi^2, and bin count."""
    angles = data[:, 0]
    intensities = data[:, 1]

    bin_width = 2.0
    min_edge = np.floor(angles.min() / bin_width) * bin_width
    max_edge = np.ceil(angles.max() / bin_width) * bin_width
    bin_edges = np.arange(min_edge, max_edge + bin_width, bin_width)
    bin_indices = np.digitize(angles, bin_edges) - 1

    bin_centers: list[float] = []
    bin_means: list[float] = []
    bin_stds: list[float] = []

    for idx in range(len(bin_edges) - 1):
        mask = bin_indices == idx
        if not np.any(mask):
            continue
        bin_centers.append((bin_edges[idx] + bin_edges[idx + 1]) / 2.0)
        bin_means.append(float(np.mean(intensities[mask])))
        bin_stds.append(float(np.std(intensities[mask], ddof=1)) if mask.sum() > 1 else 0.0)

    bin_centers_arr = np.asarray(bin_centers)
    bin_means_arr = np.asarray(bin_means)
    bin_stds_arr = np.asarray(bin_stds)

    theta_rad = np.radians(bin_centers_arr)
    design = np.column_stack(
        (
            np.sin(theta_rad),
            np.cos(theta_rad),
            np.ones_like(theta_rad),
        )
    )
    coeffs, *_ = np.linalg.lstsq(design, bin_means_arr, rcond=None)
    fitted = design @ coeffs
    residuals = bin_means_arr - fitted

    mse = float(np.mean(residuals**2)) if residuals.size else float("nan")
    nonzero_std_mask = bin_stds_arr > 0
    if np.any(nonzero_std_mask):
        chi2 = float(np.sum((residuals[nonzero_std_mask] / bin_stds_arr[nonzero_std_mask]) ** 2))
    else:
        chi2 = float("nan")

    fig, (ax_data, ax_resid) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax_data.scatter(angles, intensities, color="0.6", s=12, alpha=0.45, label="Raw measurements")
    ax_data.plot(bin_centers_arr, bin_means_arr, color="tab:blue", linewidth=1.2, label="2° bin average")
    ax_data.fill_between(
        bin_centers_arr,
        bin_means_arr - bin_stds_arr,
        bin_means_arr + bin_stds_arr,
        color="tab:blue",
        alpha=0.2,
        label="±1σ",
    )
    ax_data.plot(bin_centers_arr, fitted, color="tab:orange", linewidth=1.2, label="Sinusoidal fit")

    ax_data.set_ylabel("Light Intensity (volts)")
    ax_data.set_title(f"Brewster Angle Experiment {label}")
    ax_data.grid(True, linestyle="--", alpha=0.4)
    handles, legend_labels = ax_data.get_legend_handles_labels()
    ax_data.legend(handles, legend_labels, loc="best")

    ax_resid.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax_resid.scatter(bin_centers_arr, residuals, color="tab:purple", s=30, label="Residuals")
    ax_resid.plot(bin_centers_arr, residuals, color="tab:purple", linewidth=0.9, alpha=0.6)
    ax_resid.set_xlabel("Sensor Angle (degrees)")
    ax_resid.set_ylabel("Residual (V)")
    ax_resid.grid(True, linestyle="--", alpha=0.4)
    ax_resid.legend(loc="best")

    stats_lines = [f"MSE = {mse:.3g}" if np.isfinite(mse) else "MSE = n/a"]
    stats_lines.append(f"chi^2 = {chi2:.3g}" if np.isfinite(chi2) else "chi^2 = n/a")
    ax_resid.text(
        0.02,
        0.95,
        "\n".join(stats_lines),
        transform=ax_resid.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, linewidth=0.0),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return mse, chi2, int(bin_means_arr.size)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_files = sorted(base_dir.glob("polabbrewster*.txt"))

    for data_file in data_files:
        label, data = load_brewster_dataset(data_file)
        output_name = data_file.with_suffix(".png")
        mse, chi2, bin_count = plot_brewster_experiment(data, label, output_name)
        print(
            f"Saved plot for {data_file.name} → {output_name.name} "
            f"(min at index {int(np.argmin(data[:, 1]))}, max at index {int(np.argmax(data[:, 1]))})"
        )
        stats_msg = f"  Stats: bins={bin_count}"
        stats_msg += f", MSE={mse:.5g}" if np.isfinite(mse) else ", MSE=n/a"
        stats_msg += f", chi^2={chi2:.5g}" if np.isfinite(chi2) else ", chi^2=n/a"
        print(stats_msg)


if __name__ == "__main__":
    main()
