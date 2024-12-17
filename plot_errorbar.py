# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastcan",
#     "matplotlib",
#     "nonlinear-benchmarks",
# ]
#
# [tool.uv.sources]
# fastcan = { git = "https://github.com/scikit-learn-contrib/fastcan" }
# ///

import numpy as np
from sklearn.metrics import r2_score
import nonlinear_benchmarks
import matplotlib.pyplot as plt

from utils import fastcan_pruned_narx, random_pruned_narx, get_narx_terms


def _plot_errorbar(u, y, figure_name):
    poly_terms, narx = get_narx_terms(u, y)

    sample_step = 2000
    n_random = 10
    n_steps = 10
    r2_fastcan = np.zeros((n_random, n_steps))
    r2_random = np.zeros((n_random, n_steps))
    for i in range(n_random):
        print(figure_name, "   ", f"Random test: {i+1}/{n_random}")
        for j in range(n_steps):
            coef, intercept = fastcan_pruned_narx(
                poly_terms, y, (j + 1) * sample_step, i
            )
            r2_fastcan[i, j] = r2_score(
                np.r_[coef, intercept], np.r_[narx.coef_, narx.intercept_]
            )

            coef, intercept = random_pruned_narx(
                poly_terms, y, (j + 1) * sample_step, i
            )
            r2_random[i, j] = r2_score(
                np.r_[coef, intercept], np.r_[narx.coef_, narx.intercept_]
            )

    x = np.linspace(2000, 20000, 10)
    plt.errorbar(x, r2_fastcan.mean(axis=0), yerr=r2_fastcan.std(axis=0) / 3)
    plt.errorbar(x, r2_random.mean(axis=0), yerr=r2_random.std(axis=0) / 3)
    plt.legend(["FastCan", "Random"])
    plt.ylabel("R2")
    plt.xlabel("Number of Samples")
    plt.xticks(x)
    plt.savefig(figure_name, bbox_inches="tight")
    plt.close()


def main() -> None:
    train_val, _ = nonlinear_benchmarks.EMPS()
    train_val_u, train_val_y = train_val
    _plot_errorbar(train_val_u, train_val_y, "errorbar_emps.png")

    train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
    train_val_u, train_val_y = train_val
    _plot_errorbar(train_val_u, train_val_y, "errorbar_whbm.png")


if __name__ == "__main__":
    main()
