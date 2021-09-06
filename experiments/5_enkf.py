import itertools
import pathlib

import jax
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import tornadox

MUS = [10.0, 100.0, 1000.0]  # [10.0, 100.0, 1000.0]
NUS = [4]  # [3, 4, 5]
TOLERANCES = [10.0 ** (-p) for p in range(3, 9)]
ENSEMBLE_SIZES = [10 ** p for p in range(1, 5)]
SEEDS = [1]  # [1, 42, 54321]


results = {
    "stiffness_constant": [],
    "num_derivatives": [],
    "tol": [],
    "ensemble_size": [],
    "n_steps": [],
    "success": [],
}

combinations = list(itertools.product(MUS, NUS, TOLERANCES, ENSEMBLE_SIZES, SEEDS))
num_combinations = len(combinations)

for i_comb, (
    stiffness_constant,
    num_derivatives,
    tol,
    ensemble_size,
    seed,
) in enumerate(combinations):

    vdp = tornadox.ivp.vanderpol(
        t0=0.0, tmax=float(stiffness_constant), stiffness_constant=stiffness_constant
    )
    steprule = tornadox.step.AdaptiveSteps(abstol=tol, reltol=tol)
    ek1 = tornadox.enkf.EnK1(
        num_derivatives=num_derivatives,
        steprule=steprule,
        ensemble_size=ensemble_size,
        prng_key=jax.random.PRNGKey(seed),
    )
    solution_generator = ek1.solution_generator(vdp)

    success = True
    state, info = None, None
    try:
        for state, info in solution_generator:
            pass

    except ValueError:
        print("Interrupted after not finding good step width")
        success = False

    n_steps = info["num_steps"]

    results["stiffness_constant"].append(stiffness_constant)
    results["num_derivatives"].append(num_derivatives)
    results["tol"].append(tol)
    results["ensemble_size"].append(ensemble_size)
    results["n_steps"].append(n_steps)
    results["success"].append(int(success))

    print(
        f"+++++ {'Finished' if success else 'Interrupted'} experiment {i_comb + 1}/{num_combinations}"
    )

df = pd.DataFrame(results)

RESULT_DIR = pathlib.Path("./results/5_enkf")
if not RESULT_DIR.is_dir():
    RESULT_DIR.mkdir(parents=True)

RESULT_FILE = RESULT_DIR / "results.csv"

df.to_csv(
    RESULT_FILE, sep=";", index=False
)  # Load with pd.read_csv(file_path, sep=";")
