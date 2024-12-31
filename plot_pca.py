"""Generate PCA results"""

import click
import matplotlib.pyplot as plt
import nonlinear_benchmarks
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from fastcan import minibatch

from utils import get_dual_stable_equilibria_data, get_narx_terms


def _plot_pca(u, y, n_sample, figure_name):
    poly_terms, y, _ = get_narx_terms(u, y)
    pca = PCA(2).fit(poly_terms)
    pcs_all = pca.transform(poly_terms)

    kmeans = MiniBatchKMeans(
        n_clusters=30,
        random_state=0,
        batch_size=6,
        n_init="auto",
    ).fit(poly_terms)
    atoms = kmeans.cluster_centers_
    pcs_atoms = pca.transform(atoms)

    ids_fastcan = minibatch(poly_terms.T, atoms.T, n_sample, batch_size=5)
    pcs_fastcan = pca.transform(poly_terms[ids_fastcan])

    rng = np.random.default_rng(123)
    ids_random = rng.choice(y.size, n_sample, replace=False)
    pcs_random = pca.transform(poly_terms[ids_random])

    plt.scatter(pcs_all[:, 0], pcs_all[:, 1], s=5)
    plt.scatter(pcs_fastcan[:, 0], pcs_fastcan[:, 1], s=50, marker="o", alpha=0.9)
    plt.scatter(pcs_random[:, 0], pcs_random[:, 1], s=30, marker="*", alpha=0.9)
    plt.scatter(pcs_atoms[:, 0], pcs_atoms[:, 1], s=100, marker="+", alpha=0.9)
    plt.xlabel("The First Principle Component")
    plt.ylabel("The Second Principle Component")
    plt.legend(["All data", "FastCan pruned", "Random pruned", "Atoms"])
    plt.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")


@click.command()
@click.option("--dataset", default="dsed", help="Choose dataset from: dsed, emps, whbm")
def main(dataset) -> None:
    match dataset:
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_pca(train_val_u, train_val_y, 100, "pca_dsed.png")
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            _plot_pca(train_val_u, train_val_y, 100, "pca_emps.png")
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_pca(train_val_u, train_val_y, 100, "pca_whbm.png")
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()
