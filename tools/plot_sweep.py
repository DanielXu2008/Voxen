import csv, sys
from pathlib import Path
import matplotlib.pyplot as plt

default_path = Path(__file__).resolve().parent.parent / "build" / "sweep.csv"
path = sys.argv[1] if len(sys.argv) > 1 else default_path

with open(path) as f:
    rows = list(csv.DictReader(f))

freq = [float(r["freq"]) for r in rows]
bin_err = [float(r["binCents"]) for r in rows]
interp_err = [float(r["interpCents"]) for r in rows]

plt.figure(figsize=(10, 4.5))
plt.plot(freq, bin_err, label="Bin only", alpha=0.8)
plt.plot(freq, interp_err, label="Parabolic interpolated", linewidth=1.8)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Pitch Error (cents)")
plt.title("Pitch Estimation Error Across Singing Range (100–1000 Hz)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
