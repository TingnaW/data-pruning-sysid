"""Generate PCA results"""

import click
import matplotlib.pyplot as plt
import nonlinear_benchmarks
import numpy as np
from fastcan import minibatch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from utils import (
    get_dsed_eq,
    get_dsed_tr,
    get_dual_stable_equilibria_data,
    get_narx_terms,
)


def _plot_pca(
    u,
    y,
    n_clusters,
    batch_size,
    n_sample,
    figure_name,
    random_state,
    intercept=True,
    max_delay=10,
    split_point=None,
    split_labels=None,
):
    poly_terms, y, _ = get_narx_terms(u, y, intercept, max_delay)
    pca = PCA(2).fit(poly_terms)
    pcs_all = pca.transform(poly_terms)

    kmeans = MiniBatchKMeans(
        # n_clusters=30,
        n_clusters=n_clusters,
        random_state=random_state,
        # batch_size=6,
        batch_size=batch_size,
        n_init="auto",
    ).fit(poly_terms)
    atoms = kmeans.cluster_centers_
    pcs_atoms = pca.transform(atoms)

    ids_fastcan = minibatch(poly_terms.T, atoms.T, n_sample, batch_size=batch_size)
    pcs_fastcan = pca.transform(poly_terms[ids_fastcan])

    rng = np.random.default_rng(random_state)
    ids_random = rng.choice(y.size, n_sample, replace=False)
    pcs_random = pca.transform(poly_terms[ids_random])

    if split_labels is None:
        split_labels = ["All data"]

    if split_point is None:
        plt.scatter(pcs_all[:, 0], pcs_all[:, 1], s=50)
    else:
        plt.scatter(
            pcs_all[split_point, 0], pcs_all[split_point, 1], s=50, c="tab:blue"
        )
        plt.scatter(
            pcs_all[~split_point, 0], pcs_all[~split_point, 1], s=50, c="tab:purple"
        )
    plt.scatter(
        pcs_fastcan[:, 0], pcs_fastcan[:, 1], s=5, marker="o", alpha=0.9, c="tab:orange"
    )
    plt.scatter(
        pcs_random[:, 0], pcs_random[:, 1], s=30, marker="*", alpha=0.9, c="tab:green"
    )
    plt.scatter(
        pcs_atoms[:, 0], pcs_atoms[:, 1], s=100, marker="+", alpha=0.9, c="tab:red"
    )
    plt.xlabel("The First Principle Component")
    plt.ylabel("The Second Principle Component")
    plt.legend(split_labels + ["FastCan pruned", "Random pruned", "Atoms"])
    plt.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")


@click.command()
@click.option(
    "--dataset",
    default="dsed",
    help="Choose dataset from: dsed, emps, whbm, dsed-eq, dsed-tr",
)
@click.option("--random_state", default=0, help="Random state (int)")
def main(dataset, random_state) -> None:
    match dataset:
        case "dsed-eq":
            u, y = get_dsed_eq()
            max_delay = 3
            mask = np.arange((100 - max_delay) * 100) < (100 - max_delay) * 98
            _plot_pca(
                u,
                y,
                20,
                1000000,
                100,
                "pca_dsed_eq.png",
                random_state,
                max_delay=max_delay,
                split_point=mask,
                split_labels=["Right equil.", "Left equil."],
            )
        case "dsed-tr":
            u, y = get_dsed_tr()
            max_delay = 3
            mask = np.arange((1000 - max_delay) * 2)
            mask = (mask < 100) | (
                (mask > (1000 - max_delay)) & (mask < (1000 - max_delay) + 100)
            )
            _plot_pca(
                u,
                y,
                10,
                1000000,
                100,
                "pca_dsed_tr.png",
                random_state,
                max_delay=max_delay,
                split_point=mask,
                split_labels=["TS", "SS"],
            )
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_pca(
                train_val_u,
                train_val_y,
                20,
                1000000,
                100,
                "pca_dsed.png",
                random_state,
                max_delay=3,
            )
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            _plot_pca(
                train_val_u,
                train_val_y,
                5,
                1000000,
                100,
                "pca_emps.png",
                random_state,
                max_delay=6,
            )
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_pca(
                train_val_u,
                train_val_y,
                5,
                1000000,
                100,
                "pca_whbm.png",
                random_state,
                max_delay=10,
            )
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()
