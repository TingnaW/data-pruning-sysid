"""Generate phase portraits results for dual stable equilibria data (dsed)"""

import click
import matplotlib.pyplot as plt
import numpy as np

from utils import autonomous_dual_stable_equilibria, get_dual_stable_equilibria_data


@click.command()
@click.option("--stream", default=False, help="Use stream plot")
def main(stream: bool = False):
    figure_name = "pp_dsed.png"

    if stream:
        y, dot_y = np.meshgrid(np.linspace(-2.5, 1.5, 30), np.linspace(-1.5, 1.5, 30))
        sol = autonomous_dual_stable_equilibria((y, dot_y))
        ddot_y = sol[1]
        plt.figure()
        plt.streamplot(y, dot_y, dot_y, ddot_y, color="tab:blue", density=1.2)
    else:
        _, _, sol = get_dual_stable_equilibria_data(auto=True)
        n_init = len(sol)
        plt.figure()
        for i in range(n_init):
            plt.plot(sol[i, :, 0], sol[i, :, 1], c="tab:blue")
    fonts = 13
    plt.xlabel("y(t)", fontsize=fonts)
    plt.ylabel("dy/dt(t)", fontsize=fonts)
    plt.tick_params(axis="both", labelsize=fonts)
    plt.savefig(figure_name, bbox_inches="tight")
    print("Image " + figure_name + " has been generated.")


if __name__ == "__main__":
    main()
