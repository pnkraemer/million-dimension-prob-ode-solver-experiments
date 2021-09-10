import pathlib

import matplotlib.pyplot as plt
import pandas as pd


path = pathlib.Path("./results/vdp_stiffness_comparison/results.csv")
df = pd.read_csv(path, sep=";")


SOLVERS_TO_PLOT = [
    "ReferenceEK0",
    "KroneckerEK0",
    "DiagonalEK0",
    "ReferenceEK1",
    "DiagonalEK1",
    "ETruncationEK1",
]


# Plot the number of steps for different stiffnesses
for s in SOLVERS_TO_PLOT:
    plt.plot(df.mu, df[f"{s}_nsteps"], "o-", label=s)
plt.yscale("log")
plt.xscale("log")
plt.ylabel("Number of steps")
plt.xlabel("Stiffness constant")
plt.legend()
# plt.savefig("stability.png")
plt.show()


# Plot the error for different stiffnesses
for s in SOLVERS_TO_PLOT:
    plt.plot(df.mu, df[f"{s}_errors"], "o-", label=s)
plt.yscale("log")
plt.xscale("log")
plt.ylabel("Error")
plt.xlabel("Stiffness constant")
plt.legend()
# plt.savefig("errors.png")
plt.show()


# Plot seconds
for s in SOLVERS_TO_PLOT:
    plt.plot(df.mu, df[f"{s}_seconds"], "o-", label=s)
plt.yscale("log")
plt.xscale("log")
plt.ylabel("Seconds")
plt.xlabel("Stiffness constant")
plt.legend()
# plt.savefig("errors.png")
plt.show()
