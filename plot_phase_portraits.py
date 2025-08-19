"""Generate phase portraits results for dual stable equilibria data (dsed)"""

import matplotlib.pyplot as plt
import numpy as np

from utils import autonomous_dual_stable_equilibria, get_dual_stable_equilibria_data


def main():
    figure_name = "pp_dsed.png"

    _, _, sol = get_dual_stable_equilibria_data(auto=True)
    n_init = len(sol)
    plt.figure()
    for i in range(n_init):
        plt.plot(sol[i, :, 0], sol[i, :, 1], c="tab:blue")

    y_min = np.nanmin(sol[:, :, 0])
    y_max = np.nanmax(sol[:, :, 0])
    dot_y_min = np.nanmin(sol[:, :, 1])
    dot_y_max = np.nanmax(sol[:, :, 1])
    y, dot_y = np.meshgrid(
        np.linspace(y_min, y_max, 30), np.linspace(dot_y_min, dot_y_max, 30)
    )
    sol = autonomous_dual_stable_equilibria((y, dot_y))
    ddot_y = sol[1]
    plt.streamplot(
        y,
        dot_y,
        dot_y,
        ddot_y,
        color="tab:blue",
        density=1.2,
        minlength=0.02,
        maxlength=0.1,
        linewidth=0.5,
        arrowsize=0.5,
    )
    fonts = 13
    plt.xlabel("y(t)", fontsize=fonts)
    plt.ylabel("dy/dt(t)", fontsize=fonts)
    offset = 0.02
    plt.xlim(y_min - offset, y_max + offset)
    plt.ylim(dot_y_min - offset, dot_y_max + offset)
    plt.tick_params(axis="both", labelsize=fonts)
    plt.savefig(figure_name, bbox_inches="tight")
    print("Image " + figure_name + " has been generated.")


if __name__ == "__main__":
    main()
