import csv, sys
from pathlib import Path
import matplotlib.pyplot as plt

default_path = Path(__file__).resolve().parent.parent / "build" / "spectrum.csv"
path = sys.argv[1] if len(sys.argv) > 1 else default_path

with open(path) as f:
    rows = list(csv.DictReader(f))
freq = [float(r["freq"]) for r in rows]
mag = [float(r["mag"]) for r in rows]

plt.plot(freq, mag)
plt.xlim(0, 2000)          # zoom to the musical range
plt.yscale("log")          
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X[k]|")
plt.show()
