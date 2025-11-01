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


def plot_experiment(data: np.ndarray, label: str, output_path: Path) -> None:
    """Create a single angle-versus-intensity plot and persist it to disk."""
    angles_deg = np.degrees(data[:, 0])
    intensities = data[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles_deg, intensities, marker="o", markersize=3, linewidth=1.2)
    ax.set_xlabel("Polarizer Angle (degrees)")
    ax.set_ylabel("Light Intensity (volts)")
    ax.set_title(f"Malus Law Experiment {label}")
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

    for data_file in data_files:
        label, data = load_malus_dataset(data_file)
        output_name = data_file.with_suffix(".png")
        plot_experiment(data, label, output_name)
        print(f"Saved plot for {data_file.name} → {output_name.name}")


if __name__ == "__main__":
    main()
