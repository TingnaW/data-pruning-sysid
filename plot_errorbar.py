"""Generate error bar plot"""

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


def _plot_errorbar(
    u,
    y,
    n_sample_lower,
    n_sample_upper,
    n_steps,
    n_atoms,
    twinx,
    figure_name,
    intercept=True,
    n_random=10,
    max_delay=10,
):
    poly_terms, y, narx = get_narx_terms(u, y, intercept, max_delay)
    # n_original = u.size + y.size
    n_samples_list = np.linspace(n_sample_lower, n_sample_upper, n_steps, endpoint=True)
    n_original = (u.size + y.size)-(max_delay+1)+1

    density_percentage = []
    sample_step = int((n_sample_upper - n_sample_lower) / (n_steps - 1))
    r2_fastcan = np.zeros((n_steps,n_random))
    r2_random = np.zeros((n_steps,n_random))
    columns = [*Progress.get_default_columns()]
    columns[-1] = TimeRemainingColumn(elapsed_when_finished=True)
    with Progress(*columns, auto_refresh=True) as pb:
        t1 = pb.add_task("[red]Pruning data...", total=n_steps)
        t2 = pb.add_task("[green]Random test...", total=n_random)
        for i in range(n_steps):
            n_samples = n_samples_list[i]
            # density_level = (poly_terms.shape[1] + 1) * n_samples / n_original
            density_level = n_samples / n_original
            density_percentage.append(density_level*100)
            for j in range(n_random):
                coef = fastcan_pruned_narx(
                    poly_terms,
                    y,
                    i * sample_step + n_sample_lower,
                    j,
                    n_atoms=n_atoms,
                    intercept=intercept,
                )
                r2_fastcan[i, j] = get_r2(coef, narx)

                coef = random_pruned_narx(
                    poly_terms,
                    y,
                    i * sample_step + n_sample_lower,
                    j,
                    intercept=intercept,
                )
                r2_random[i, j] = get_r2(coef, narx)
                pb.update(task_id=t1, completed=j + 1)
            pb.update(task_id=t2, completed=i + 1)

    # x = np.linspace(n_sample_lower, n_sample_upper, n_steps, endpoint=True) 

    density_percentage = np.array(density_percentage)
    fonts = 14
    fig, ax1 = plt.subplots()
    if twinx:
        ax2 = ax1.twinx()
        ax2.errorbar(
            n_samples_list,
            r2_random.mean(axis=1),
            yerr=r2_random.std(axis=1) / np.sqrt(n_random),
            color="tab:orange",
            label="Random",
        )
        ax2.set_ylabel("R-squared", fontsize=fonts)
        ax2.tick_params(axis="y", labelsize=fonts)
    else:
        ax1.errorbar(
            n_samples_list,
            r2_random.mean(axis=1),
            yerr=r2_random.std(axis=1) / np.sqrt(n_random),
            color="tab:orange",
            label="Random",
        )
    ax1.errorbar(
        n_samples_list,
        r2_fastcan.mean(axis=1),
        yerr=r2_fastcan.std(axis=1) / np.sqrt(n_random),
        color="tab:blue",
        label="FastCan",
    )
    
    ax1.set_ylabel("R-squared", fontsize=fonts)
    ax1.set_xlabel("Number of selected samples", fontsize=fonts)
    ax1.set_xticks(n_samples_list)
    ax1.tick_params(axis='x', labelsize=fonts)
    ax1.tick_params(axis='y', labelsize=fonts)

    fig.legend(loc="lower right", bbox_to_anchor=(0.88, 0.12), fontsize=fonts)

    # Add twin x-axis: convert cm to inches
    ax_top = ax1.twiny()
    ax_top.set_xlim(ax1.get_xlim())  # Match bottom axis range
    ax_top.set_xticks(n_samples_list)
    # ax_top.set_xticklabels([f"{p:.2f}" for p in density_percentage])
    labels = [f"{p:.2f}" if i % 2 == 0 else "" for i, p in enumerate(density_percentage)]
    ax_top.set_xticklabels(labels)
    ax_top.set_xlabel("Percentage of selected samples (%)", fontsize=fonts)
    ax_top.tick_params(axis='x', labelsize=fonts)

    fig.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")


@click.command()
@click.option(
    "--dataset",
    default="dsed",
    help="Choose dataset from: dsed, emps, whbm, dsed-eq, dsed-tr",
)
@click.option("--n_random", default=10, help="Set the number of random tests")
def main(dataset, n_random) -> None:
    match dataset:
        case "dsed-eq":
            u, y = get_dsed_eq()
            _plot_errorbar(
                u,
                y,
                50,
                150,
                11,
                20,
                False,
                "errorbar_dsed_eq.png",
                n_random=n_random,
                max_delay=4,
            )
        case "dsed-tr":
            u, y = get_dsed_tr()
            _plot_errorbar(
                u,
                y,
                50,
                150,
                11,
                20,
                False,
                "errorbar_dsed_tr.png",
                n_random=n_random,
                max_delay=6,
            )
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_errorbar(
                train_val_u,
                train_val_y,
                50,
                150,
                11,
                15,
                False,
                "errorbar_dsed.png",
                n_random=n_random,
                max_delay=4,
            )
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            # No intercept for EMPS dataset
            _plot_errorbar(
                train_val_u,
                train_val_y,
                20,
                120,
                11,
                25,  # No atoms for EMPS dataset
                False,
                "errorbar_emps.png",
                # intercept=True,
                n_random=n_random,
                max_delay=4,
            )
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_errorbar(
                train_val_u,
                train_val_y,
                20,
                120,
                11,
                5,  # No atoms for WHBM dataset
                False,
                "errorbar_whbm.png",
                n_random=n_random,
                max_delay=7,
            )
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()
