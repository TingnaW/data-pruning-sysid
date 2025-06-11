"""Print terms of NARX models."""

import click
import nonlinear_benchmarks
from fastcan.narx import make_narx, print_narx
from fastcan.utils import mask_missing_values
from sklearn.metrics import r2_score

from utils import get_dsed_eq, get_dsed_tr, get_dual_stable_equilibria_data


@click.command()
@click.option(
    "--dataset",
    default="dsed",
    help="Choose dataset from: dsed, emps, whbm, dsed-eq, dsed-tr",
)
@click.option("--nterms", default=10, help="Number of NARX terms (int)")
@click.option("--delay", default=10, help="Maximum delay (int)")
@click.option("--poly", default=3, help="Maximum polynomial degree of NARX terms (int)")
@click.option(
    "--intercept", default=True, help="Include intercept in NARX model (bool)"
)
def main(dataset, nterms, delay, poly, intercept) -> None:
    match dataset:
        case "dsed-eq":
            u, y = get_dsed_eq()
            u = u.reshape(-1, 1)
        case "dsed-tr":
            u, y = get_dsed_tr()
            u = u.reshape(-1, 1)
        case "dsed":
            u, y, _ = get_dual_stable_equilibria_data()
            u = u.reshape(-1, 1)
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            u, y = train_val
            u = u.reshape(-1, 1)
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            u, y = train_val
            u = u.reshape(-1, 1)
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )
    narx = make_narx(
        u,
        y,
        n_terms_to_select=nterms,
        max_delay=delay,
        poly_degree=poly,
        fit_intercept=intercept,
        verbose=0,
    ).fit(
        u,
        y,
    )
    y_hat = narx.predict(u, y_init=y)
    y, y_hat = mask_missing_values(y, y_hat)
    print("Training R2 score:", r2_score(y, y_hat))
    print_narx(narx)


if __name__ == "__main__":
    main()
