from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_malus_dataset(path: Path) -> Tuple[str, np.ndarray]:
    """Return the experiment label and numeric data from a Malus law text file."""
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        raw_title = handle.readline().strip()
    data = np.loadtxt(path, skiprows=2)
    label = raw_title.replace("Polarization of Light 'theta' Data File", "").strip()
    return label, data


def bin_by_degree(angles_deg: np.ndarray, intensities: np.ndarray, bin_width: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate intensity measurements into degree-wide bins."""
    if angles_deg.size == 0:
        raise ValueError("No angle measurements provided for binning.")

    low_edge = np.floor(angles_deg.min())
    high_edge = np.ceil(angles_deg.max())
    edges = np.arange(low_edge, high_edge + bin_width + 1e-9, bin_width)
    bin_indices = np.digitize(angles_deg, edges) - 1

    mean_angles: List[float] = []
    mean_intensities: List[float] = []
    std_intensities: List[float] = []

    for idx in range(len(edges) - 1):
        mask = bin_indices == idx
        if not np.any(mask):
            continue
        mean_angles.append(float(angles_deg[mask].mean()))
        mean_intensities.append(float(intensities[mask].mean()))
        if mask.sum() > 1:
            std_intensities.append(float(intensities[mask].std(ddof=1)))
        else:
            std_intensities.append(0.0)

    return (
        np.asarray(mean_angles),
        np.asarray(mean_intensities),
        np.asarray(std_intensities),
    )


def _describe_degree(degree: int) -> str:
    descriptions = {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic"}
    return descriptions.get(degree, f"Degree {degree}")


def _safe_sigma(std_values: np.ndarray, fallback_scale: float) -> np.ndarray:
    nonzero = std_values[std_values > 0]
    if nonzero.size == 0:
        scale = fallback_scale if fallback_scale > 0 else 1.0
        return np.full_like(std_values, scale, dtype=float)
    scale = nonzero.mean()
    sigma = np.where(std_values > 0, std_values, scale)
    sigma[sigma <= 0] = scale if scale > 0 else 1.0
    return sigma.astype(float)


def fit_cos_polynomial_and_plot(
    angles_deg_raw: np.ndarray,
    intensities_raw: np.ndarray,
    mean_angles_deg: np.ndarray,
    mean_intensity: np.ndarray,
    sigma: np.ndarray,
    degree: int,
    angle_offset: float,
    bin_width: float,
    title: str,
    output_path: Path,
) -> Dict[str, float]:
    """Fit intensity with a polynomial in cos(θ+offset) and render angle-domain diagnostics."""
    if mean_angles_deg.size <= degree:
        raise ValueError(
            f"Not enough binned data points ({mean_angles_deg.size}) for a degree-{degree} polynomial fit."
        )

    cos_binned = np.cos(np.radians(mean_angles_deg))
    weights = 1.0 / sigma

    coefficients = np.polyfit(cos_binned, mean_intensity, degree, w=weights)
    poly = np.poly1d(coefficients)

    residuals = mean_intensity - poly(cos_binned)
    mse = float(np.mean(residuals**2))
    chi2 = float(np.sum((residuals / sigma) ** 2))
    dof = mean_angles_deg.size - (degree + 1)
    reduced_chi2 = float(chi2 / dof) if dof > 0 else float("nan")

    sort_idx = np.argsort(mean_angles_deg)
    angles_sorted = mean_angles_deg[sort_idx]
    mean_intensity_sorted = mean_intensity[sort_idx]
    sigma_sorted = sigma[sort_idx]
    residuals_sorted = residuals[sort_idx]

    dense_angles = np.linspace(angles_deg_raw.min(), angles_deg_raw.max(), 400)
    fit_values = poly(np.cos(np.radians(dense_angles)))

    fig, (ax_main, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_main.scatter(
        angles_deg_raw,
        intensities_raw,
        s=18,
        alpha=0.35,
        label="Raw samples",
        color="tab:gray",
    )
    ax_main.fill_between(
        angles_sorted,
        mean_intensity_sorted - sigma_sorted,
        mean_intensity_sorted + sigma_sorted,
        color="tab:blue",
        alpha=0.2,
        label="±1σ bin envelope",
    )
    ax_main.plot(
        angles_sorted,
        mean_intensity_sorted,
        marker="o",
        linestyle="none",
        color="tab:blue",
        label=f"{bin_width:g}° bin mean",
    )
    offset_deg = np.degrees(angle_offset)
    ax_main.plot(
        dense_angles,
        fit_values,
        color="tab:orange",
        linewidth=2,
        label=f"{_describe_degree(degree)} polynomial in cos(θ + {offset_deg:.2f}°)",
    )
    ax_main.set_ylabel("Light Intensity (volts)")
    ax_main.set_title(title)
    ax_main.grid(True, linestyle="--", alpha=0.4)
    metrics_lines = [f"MSE: {mse:.4g}", f"χ²: {chi2:.3g}"]
    if np.isfinite(reduced_chi2):
        metrics_lines.append(f"χ²/dof: {reduced_chi2:.3g}")
    ax_main.text(
        0.02,
        0.95,
        "\n".join(metrics_lines),
        transform=ax_main.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax_main.legend(loc="best", frameon=True)

    ax_resid.fill_between(
        angles_sorted,
        -sigma_sorted,
        sigma_sorted,
        color="tab:blue",
        alpha=0.2,
    )
    ax_resid.scatter(mean_angles_deg, residuals, color="tab:orange", s=28)
    ax_resid.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax_resid.set_ylabel("Residuals")
    ax_resid.set_xlabel("Angle θ (degrees)")
    ax_resid.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return {
        "mse": mse,
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
    }


def process_dataset(
    data_file: Path,
    angle_offset: float,
    poly_degree: int,
    bin_width: float = 2.0,
) -> None:
    label, data = load_malus_dataset(data_file)
    angles_rad = data[:, 0]
    intensities = data[:, 1]

    corrected_angles_rad = angles_rad + angle_offset

    corrected_angles_deg = np.degrees(corrected_angles_rad)

    mean_angles_deg, mean_intensity, std_intensity = bin_by_degree(
        corrected_angles_deg, intensities, bin_width=bin_width
    )
    fallback_sigma = float(np.std(intensities, ddof=1)) if intensities.size > 1 else 1.0
    sigma = _safe_sigma(std_intensity, fallback_sigma)

    base_stem = data_file.stem
    angle_plot_path = data_file.with_name(f"{base_stem}_angle_fit.png")

    metrics = fit_cos_polynomial_and_plot(
        corrected_angles_deg,
        intensities,
        mean_angles_deg,
        mean_intensity,
        sigma,
        poly_degree,
        angle_offset,
        bin_width,
        f"Malus Law Experiment {label} — Intensity vs Angle",
        angle_plot_path,
    )

    print(
        f"{data_file.name} — degree-{poly_degree} polynomial in cos(θ + {angle_offset:.3g} rad): "
        f"MSE={metrics['mse']:.4g}, χ²={metrics['chi2']:.4g}, χ²/dof={metrics['reduced_chi2']:.4g}"
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_files = [
        base_dir / "polabMalus1.txt",
        base_dir / "polabMalus2.txt",
    ]

    process_dataset(data_files[0], angle_offset=0.42, poly_degree=2)
    process_dataset(data_files[1], angle_offset=-0.2, poly_degree=4)


if __name__ == "__main__":
    main()
