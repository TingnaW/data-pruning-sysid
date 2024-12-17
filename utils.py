"""Utils functions."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import MiniBatchKMeans

from fastcan import minibatch
from fastcan.narx import make_time_shift_features, make_poly_features, make_narx


def get_narx_terms(u, y):
    """
    Generate NARX terms from input and output data.

    Parameters
    ----------
    u : array-like
        Input data.

    y : array-like
        Output data.

    Returns
    -------
    poly_terms : array-like
        Polynomial terms generated from the input and output data.

    narx : object
        Fitted NARX model.
    """

    narx = make_narx(
        u.reshape(-1, 1),
        y,
        n_features_to_select=10,
        max_delay=10,
        poly_degree=3,
        verbose=0,
    ).fit(
        u.reshape(-1, 1),
        y,
    )

    xy_hstack = np.c_[u, y]
    time_shift_vars = make_time_shift_features(xy_hstack, narx.time_shift_ids_)
    poly_terms = make_poly_features(time_shift_vars, narx.poly_ids_)
    return poly_terms, narx


def fastcan_pruned_narx(terms, y, n_samples_to_select: int, random_state: int):
    """
    Fit a NARX model with data pruned by FastCan.

    Parameters
    ----------
    terms : array-like
        Input data.

    y : array-like
        Output data.

    n_samples_to_select : int
        Number of samples to select.

    random_state : int
        Random state.

    Returns
    -------
    coef : array-like
        Coefficients of the linear regression.

    intercept : float
        Intercept of the linear regression.
    """
    kmeans = MiniBatchKMeans(
        n_clusters=100,
        random_state=random_state,
        batch_size=6,
        n_init="auto",
    ).fit(terms)
    atoms = kmeans.cluster_centers_
    ids_fastcan = minibatch(terms.T, atoms.T, n_samples_to_select, batch_size=7)
    pruned_narx = LinearRegression().fit(terms[ids_fastcan], y[ids_fastcan])
    return pruned_narx.coef_, pruned_narx.intercept_


def random_pruned_narx(terms, y, n_samples_to_select: int, random_state: int):
    """
    Fit a NARX model with data pruned by random selection.

    Parameters
    ----------
    terms : array-like
        Input data.

    y : array-like
        Output data.

    n_samples_to_select : int
        Number of samples to select.

    random_state : int
        Random state.

    Returns
    -------
    coef : array-like
        Coefficients of the linear regression.

    intercept : float
        Intercept of the linear regression.
    """
    rng = np.random.default_rng(random_state)
    ids_random = rng.choice(y.size, n_samples_to_select, replace=False)
    pruned_narx = LinearRegression().fit(terms[ids_random], y[ids_random])
    return pruned_narx.coef_, pruned_narx.intercept_
