import pathlib
import jax.numpy as jnp
import matplotlib.pyplot as plt


result_dir = pathlib.Path("./results/0_figure1_diagonalek1_pdesolution")
ts = jnp.load(result_dir / "times.npy")
y_means = jnp.load(result_dir / "means.npy")
y_stds = jnp.load(result_dir / "stddevs.npy")

cmap_means, cmap_stds = "coolwarm", "Purples"

# Visualize
d = D // 2
_d = int(d ** (1 / 2))
# Plot means
for _t, _y in zip(ts, y_means):
    fig = plt.figure()
    cm = plt.imshow(_y[:d].reshape(_d, _d), cmap=cmap_means, vmin=-1, vmax=1)
    plt.axis("off")
    fig.colorbar(cm, extend="both")
    fig.tight_layout()
    fig.savefig(result_dir / f"mean_t={_t}.pdf")
# Plot stddevs
for _t, _y in zip(ts, y_stds):
    fig = plt.figure()
    cm = plt.imshow(_y[:d].reshape(_d, _d), cmap=cmap_stds, vmin=0)
    plt.axis("off")
    fig.colorbar(cm, extend="max")
    fig.tight_layout()
    fig.savefig(result_dir / f"stddev_t={_t}.pdf")
plt.close("all")
