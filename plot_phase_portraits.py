"""Generate phase portraits results for dual stable equilibria data (dsed)"""

import matplotlib.pyplot as plt

from utils import get_dual_stable_equilibria_data


def main():
    figure_name = "pp_dsed.png"
    _, _, sol = get_dual_stable_equilibria_data(auto=True)
    n_init = len(sol)
    for i in range(n_init):
        plt.plot(sol[i, :, 0], sol[i, :, 1], c="tab:blue")
    fonts =13
    plt.xlabel("y(t)", fontsize=fonts)
    plt.ylabel("dy/dt(t)",fontsize=fonts)
    plt.tick_params(axis='both', labelsize=fonts)
    plt.savefig(figure_name, bbox_inches="tight")
    print("Image " + figure_name + " has been generated.")


if __name__ == "__main__":
    main()
