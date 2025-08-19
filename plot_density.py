"""Plot R2 score vs. density level (fraction of samples remaining after pruning)"""

import click
import matplotlib.pyplot as plt
import nonlinear_benchmarks
import numpy as np
from rich.progress import Progress, TimeRemainingColumn

from utils import (
    fastcan_pruned_narx,
    get_dsed_eq,
    get_dsed_tr,
    get_dual_stable_equilibria_data,
    get_narx_terms,
    get_r2,
    random_pruned_narx,
)


def _plot_density(
    u,
    y,
    n_atoms,
    max_delay,
    n_samples_list,
    figure_name,
):
    n_random = 10
    n_levels = len(n_samples_list)
    poly_terms, y, narx = get_narx_terms(u, y, max_delay=max_delay)
    # n_original = u.size + y.size
    n_original = (u.size + y.size) - (max_delay + 1) + 1

    r2_fastcan = np.zeros((n_levels, n_random))
    r2_random = np.zeros((n_levels, n_random))
    density_percentage = []
    columns = [*Progress.get_default_columns()]
    columns[-1] = TimeRemainingColumn(elapsed_when_finished=True)
    with Progress(*columns, auto_refresh=True) as pb:
        t1 = pb.add_task("[red]Pruning data...", total=n_levels)
        t2 = pb.add_task("[green]Random test...", total=n_random)
        for i in range(n_levels):
            n_samples = n_samples_list[i]
            # density_level = (poly_terms.shape[1] + 1) * n_samples / n_original
            density_level = n_samples / n_original
            density_percentage.append(density_level * 100)
            for j in range(n_random):
                coef = fastcan_pruned_narx(poly_terms, y, n_samples, j, n_atoms=n_atoms)
                r2_fastcan[i, j] = get_r2(coef, narx)

                coef = random_pruned_narx(poly_terms, y, n_samples, j)
                r2_random[i, j] = get_r2(coef, narx)
                pb.update(task_id=t1, completed=j + 1)
            pb.update(task_id=t2, completed=i + 1)

    mean_r2_fastcan = np.mean(r2_fastcan, axis=1)
    std_r2_fastcan = np.std(r2_fastcan, axis=1)
    mean_r2_random = np.mean(r2_random, axis=1)
    std_r2_random = np.std(r2_random, axis=1)

    plt.figure()
    plt.plot(density_percentage, mean_r2_fastcan, label="FastCan mean", marker="o")
    plt.fill_between(
        density_percentage,
        mean_r2_fastcan - std_r2_fastcan,
        mean_r2_fastcan + std_r2_fastcan,
        alpha=0.2,
        label="FastCan SD",
    )

    plt.plot(density_percentage, mean_r2_random, label="Random mean", marker="x")
    plt.fill_between(
        density_percentage,
        mean_r2_random - std_r2_random,
        mean_r2_random + std_r2_random,
        alpha=0.2,
        label="Random SD",
    )

    fonts = 14
    plt.xlabel("Percentage of selected samples (%)", fontsize=fonts)
    plt.ylabel("R-squared", fontsize=fonts)
    # Tick label font size
    plt.tick_params(axis="both", labelsize=fonts)
    plt.legend(fontsize=fonts)
    plt.grid(True)

    plt.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")


@click.command()
@click.option(
    "--dataset",
    default="dsed",
    help="Choose dataset from: dsed, emps, whbm, dsed-eq, dsed-tr",
)
def main(dataset) -> None:
    match dataset:
        case "dsed-eq":
            u, y = get_dsed_eq()
            _plot_density(
                u,
                y,
                n_atoms=20,
                max_delay=4,
                n_samples_list=np.linspace(50, 150, 11, dtype=int),
                figure_name="density_dsed_eq.png",
            )
        case "dsed-tr":
            u, y = get_dsed_tr()
            _plot_density(
                u,
                y,
                n_atoms=20,
                max_delay=6,
                n_samples_list=np.linspace(50, 150, 11, dtype=int),
                figure_name="density_dsed_tr.png",
            )
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_density(
                train_val_u,
                train_val_y,
                n_atoms=15,
                max_delay=4,
                n_samples_list=np.linspace(50, 150, 11, dtype=int),
                figure_name="density_dsed.png",
            )

        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            _plot_density(
                train_val_u,
                train_val_y,
                n_atoms=25,
                max_delay=4,
                n_samples_list=np.linspace(20, 120, 11, dtype=int),
                figure_name="density_emps.png",
            )
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_density(
                train_val_u,
                train_val_y,
                n_atoms=5,
                max_delay=7,
                n_samples_list=np.linspace(20, 120, 11, dtype=int),
                figure_name="density_whbm.png",
            )


if __name__ == "__main__":
    main()
