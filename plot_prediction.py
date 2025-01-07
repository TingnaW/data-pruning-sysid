"""Visualize the prediction performance of NARX models."""

import click
import matplotlib.pyplot as plt
import nonlinear_benchmarks
from sklearn.metrics import r2_score
from fastcan.narx import make_narx, _mask_missing_value

from utils import get_dual_stable_equilibria_data


def _plot_prediction(u, y, n_terms, max_delay, poly_degree, plot_n_samples, figure_name, narx=None):
    if narx is None:
        narx = make_narx(
            u.reshape(-1, 1),
            y,
            n_features_to_select=n_terms,
            max_delay=max_delay,
            poly_degree=poly_degree,
            verbose=0,
        ).fit(
            u.reshape(-1, 1),
            y,
        )
    y_hat = narx.predict(u.reshape(-1, 1), y_init=y[:narx.max_delay_])

    plt.plot(y[:plot_n_samples], label="True")
    plt.plot(y_hat[:plot_n_samples], label="Predicted")
    plt.xlabel("Time index k")
    plt.legend()
    plt.title(f"NARX prediction results (R-squared: {r2_score(y[:plot_n_samples], y_hat[:plot_n_samples]):.5f})")
    plt.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")
    return narx


@click.command()
@click.option("--dataset", default="dsed", help="Choose dataset from: dsed, emps, whbm")
@click.option("--nterms", default=10, help="Number of NARX terms (int)")
@click.option("--delay", default=10, help="Maximum delay (int)")
@click.option("--poly", default=3, help="Maximum polynomial degree of NARX terms (int)")
def main(dataset, nterms, delay, poly) -> None:
    match dataset:
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data(y0=[[0.6, 0.8]])
            test_val_u, test_val_y, _ = get_dual_stable_equilibria_data(y0=[[-0.6, 0.8]])
            narx = _plot_prediction(train_val_u, train_val_y, nterms, delay, poly, 100, "pred_train_dsed.png")
            _plot_prediction(test_val_u, test_val_y, nterms, delay, poly, 100, "pred_test_dsed.png")
        case "emps":
            train_val, test_val = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            test_val_u, test_val_y = test_val
            narx = _plot_prediction(train_val_u, train_val_y, nterms, delay, poly, 25000, "pred_train_emps.png")
            _plot_prediction(test_val_u, test_val_y, nterms, delay, poly, 25000, "pred_test_emps.png", narx)
        case "whbm":
            train_val, test_val = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            test_val_u, test_val_y = test_val
            narx = _plot_prediction(train_val_u, train_val_y, nterms, delay, poly, 1000, "pred_train_whbm.png")
            _plot_prediction(test_val_u, test_val_y, nterms, delay, poly, 1000, "pred_test_whbm.png", narx)
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()