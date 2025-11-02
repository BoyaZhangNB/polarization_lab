from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_malus_dataset(path: Path) -> Tuple[str, np.ndarray]:
    """Return the experiment label and numeric data from a Malus law text file."""
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        raw_title = handle.readline().strip()
    data = np.loadtxt(path, skiprows=2)
    label = raw_title.replace("Polarization of Light 'theta' Data File", "").strip()
    return label, data


def plot_series(x: np.ndarray, y: np.ndarray, xlabel: str, title: str, output_path: Path) -> None:
    """Render and persist a single Malus-law diagnostic plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # scatter avoids connecting lines so the trend is visible without implying interpolation
    ax.scatter(x, y, s=18, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Light Intensity (volts)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_files = [
        base_dir / "polabMalus1.txt",
        base_dir / "polabMalus2.txt",
    ]

    for i, data_file in enumerate(data_files):
        label, data = load_malus_dataset(data_file)
        angles_rad = data[:, 0]
        intensities = data[:, 1]

        angles_deg = np.degrees(angles_rad+0.42 if i ==0 else angles_rad-0.2) # normalize by 0.35 rad
        cos_theta = np.cos(angles_rad+0.42 if i ==0 else angles_rad-0.2) # s.t. highest intensity starts at 0
        cos_sq_theta = cos_theta ** 2

        base_stem = data_file.stem

        angle_path = data_file.with_name(f"{base_stem}_angle.png")
        cos_path = data_file.with_name(f"{base_stem}_cos.png")
        cos_sq_path = data_file.with_name(f"{base_stem}_cos2.png")

        plot_series(
            angles_deg,
            intensities,
            "Polarizer Angle (degrees)",
            f"Malus Law Experiment {label} — Intensity vs Angle",
            angle_path,
        )
        plot_series(
            cos_theta,
            intensities,
            "cos(θ)",
            f"Malus Law Experiment {label} — Intensity vs cos(θ)",
            cos_path,
        )
        plot_series(
            cos_sq_theta,
            intensities,
            "cos²(θ)",
            f"Malus Law Experiment {label} — Intensity vs cos²(θ)",
            cos_sq_path,
        )

        print(
            "Saved plots for "
            f"{data_file.name} → {angle_path.name}, {cos_path.name}, {cos_sq_path.name}"
        )


if __name__ == "__main__":
    main()
