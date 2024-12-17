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


def _plot_box(u, y, figure_name):
    poly_terms, narx = get_narx_terms(u, y)

    n_samples_to_select = 10000
    n_random = 10
    r2_fastcan = np.zeros(n_random)
    r2_random = np.zeros(n_random)
    for i in range(n_random):
        print(figure_name, "   ", f"Random test: {i+1}/{n_random}")
        coef, intercept = fastcan_pruned_narx(poly_terms, y, n_samples_to_select, i)
        r2_fastcan[i] = r2_score(
            np.r_[coef, intercept], np.r_[narx.coef_, narx.intercept_]
        )

        coef, intercept = random_pruned_narx(poly_terms, y, n_samples_to_select, i)
        r2_random[i] = r2_score(
            np.r_[coef, intercept], np.r_[narx.coef_, narx.intercept_]
        )

    plt.boxplot(np.c_[r2_fastcan, r2_random])
    plt.ylabel("R2")
    plt.xticks(ticks=[1, 2], labels=["FastCan", "Random"])
    plt.savefig(figure_name, bbox_inches="tight")
    plt.close()


def main() -> None:
    train_val, _ = nonlinear_benchmarks.EMPS()
    train_val_u, train_val_y = train_val
    _plot_box(train_val_u, train_val_y, "box_emps.png")

    train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
    train_val_u, train_val_y = train_val
    _plot_box(train_val_u, train_val_y, "box_whbm.png")


if __name__ == "__main__":
    main()
